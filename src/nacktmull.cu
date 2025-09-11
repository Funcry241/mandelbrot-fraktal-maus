///// Otter: Direkte Mandelbrot-Iteration (ohne Referenz-Orbit), GT-Palette (Cyan→Amber), Smooth-Coloring.
///  Schneefuchs: API & Mapping unverändert (pixelToComplex), deterministisch.
///  Maus: Heatmap-Vertrag bleibt (innen = maxIter).
///  Bonus: Eye-Candy-Animation (sanft, monotone Farbabbildung bleibt erhalten).

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>      // float2, uchar4
#include <vector_functions.h>  // make_float2, make_float3, make_uchar4
#include <cmath>
#include <chrono>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"
#include "nacktmull_math.cuh"  // pixelToComplex(...)

// ============================================================================
// Device Utilities
// ============================================================================
__device__ __forceinline__ float  clamp01(float x)                  { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ __forceinline__ float  mixf(float a, float b, float t)   { return a + t * (b - a); }
__device__ __forceinline__ float3 mix3(float3 a, float3 b, float t) { return make_float3(mixf(a.x,b.x,t), mixf(a.y,b.y,t), mixf(a.z,b.z,t)); }

// Cardioid / Period-2-Bulb (Early-Out)
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y * y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true; // main cardioid
    const float xp = x + 1.0f;                // period-2 bulb
    if (xp*xp + y2 <= 0.0625f) return true;
    return false;
}

// sRGB <-> Linear
__device__ __forceinline__ float  srgb_to_linear(float c){
    return (c <= 0.04045f) ? (c/12.92f) : powf((c + 0.055f)/1.055f, 2.4f);
}
__device__ __forceinline__ float  linear_to_srgb(float c){
    return (c <= 0.0031308f) ? (12.92f*c) : (1.055f*powf(c, 1.0f/2.4f) - 0.055f);
}
__device__ __forceinline__ float3 srgb_to_linear3(float3 c){
    return make_float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}
__device__ __forceinline__ float3 linear_to_srgb3(float3 c){
    return make_float3(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

// ============================================================================
// GT-Palette (Cyan→Amber), Interpolation im Linearraum + sanfte Animation
// ============================================================================
__device__ __forceinline__ uchar4 gtPalette_u8(float x, bool inSet, float t){
    // Tuning (Eye-Candy, aber konservativ/monoton):
    // - gamma etwas niedriger für mehr Mittenton-Zeichnung
    // - "lift" hebt dunkle Tiefen leicht an (mehr Struktur im Außenbereich)
    // - warmShift driftet sanft über Zeit
    // - ultrafeine dynamische Mikro-Isolinien (sehr geringe Amplitude)
    const float gamma        = 0.86f;
    const float lift         = 0.07f;                         // 0.05..0.10
    const float baseVibr     = 1.04f;                         // Grundvibrance
    const float addVibrMax   = 0.06f;                         // Zusatzvibrance (x-abhängig)
    const float warmDriftAmp = 0.06f;                         // zeitl. Warmdrift ±6%
    const float warmShift    = 1.00f + warmDriftAmp * __sinf(0.30f * t);
    const float stripes      = 0.012f;                        // ultrafein
    const float stripeFreq   = 6.2f;

    if (inSet) return make_uchar4(10, 12, 16, 255);

    // Eingangsshaping
    x = clamp01(powf(clamp01(x), gamma));
    x = clamp01((x + lift) / (1.0f + lift));                  // Low-End anheben

    // Anchors (dezent heller im dunklen Bereich)
    const float  p[8] = { 0.00f, 0.12f, 0.25f, 0.42f, 0.60f, 0.78f, 0.95f, 1.00f };
    const float3 c[8] = {
        make_float3(11/255.f, 14/255.f, 26/255.f), // #0B0E1A (vorher #08090F)
        make_float3(20/255.f, 54/255.f,102/255.f), // #143666 (vorher #112D5F)
        make_float3(22/255.f, 84/255.f,159/255.f), // #16549F
        make_float3(36/255.f,178/255.f,191/255.f), // #24B2BF
        make_float3(255/255.f,210/255.f, 87/255.f),// #FFD257
        make_float3(236/255.f,121/255.f, 44/255.f),// #EC792C
        make_float3(171/255.f, 34/255.f, 61/255.f),// #AB223D
        make_float3(250/255.f,250/255.f,250/255.f) // #FAFAFA
    };

    int j = 0;
    #pragma unroll
    for (int i=0; i<7; ++i) { if (x >= p[i]) j = i; }
    const float span = fmaxf(p[j+1] - p[j], 1e-6f);
    float tseg = clamp01((x - p[j]) / span);
    tseg = tseg*tseg*(3.0f - 2.0f*tseg); // smootherstep

    float3 aLin  = srgb_to_linear3(c[j]);
    float3 bLin  = srgb_to_linear3(c[j+1]);
    float3 rgbLn = mix3(aLin, bLin, tseg);

    // Sanfte „Breathing“-Bewegung: x -> x'
    // Monotonie bleibt erhalten, Amplitude klein.
    {
        const float breath = 0.08f * __sinf(0.80f * t);       // ±0.08
        const float xprime = clamp01(x + breath * x * (1.0f - x));
        // leicht auf Sättigung wirken (x-abhängig)
        const float vibr = baseVibr + addVibrMax * clamp01((xprime - 0.10f) * (1.0f / 0.40f));
        const float luma = 0.2126f*rgbLn.x + 0.7152f*rgbLn.y + 0.0722f*rgbLn.z;
        rgbLn = make_float3(
            luma + (rgbLn.x - luma) * vibr * warmShift,
            luma + (rgbLn.y - luma) * vibr * 1.00f,
            luma + (rgbLn.z - luma) * vibr * (2.0f - warmShift)
        );
    }

    // Ultraf eine dynamische Mikro-Isolinien (nur mittlere Tonwerte, minimal)
    {
        const float mid = 4.0f * x * (1.0f - x);              // 0..1, peak bei x=0.5
        const float amp = stripes * mid;                      // nur Mitte betonen
        if (amp > 0.0f){
            const float phase = 6.2831853f * (x * stripeFreq + 0.10f * t);
            const float s = 0.5f + 0.5f * __sinf(phase);
            const float boost = 1.0f + amp * (s*s*s*s);       // Highlights biasen
            rgbLn.x *= boost; rgbLn.y *= boost; rgbLn.z *= boost;
        }
    }

    const float3 srgb = linear_to_srgb3(make_float3(
        clamp01(rgbLn.x), clamp01(rgbLn.y), clamp01(rgbLn.z)
    ));

    return make_uchar4(
        (unsigned char)(255.0f*clamp01(srgb.x) + 0.5f),
        (unsigned char)(255.0f*clamp01(srgb.y) + 0.5f),
        (unsigned char)(255.0f*clamp01(srgb.z) + 0.5f),
        255
    );
}

// Smooth-Iterations → Farbwert (+ Edge-Glow & Animation)
__device__ __forceinline__ uchar4 gtColor_fromSmoothState(int it, int maxIterations, float zx, float zy, float t){
    const bool inSet = (it >= maxIterations);
    if (inSet) return gtPalette_u8(0.0f, true, t);

    const float mag2 = zx*zx + zy*zy;
    if (mag2 > 1.0000001f && it > 0){
        const float mag = sqrtf(mag2);
        const float l2  = __log2f(__log2f(mag));
        float x = ((float)it - l2) / (float)maxIterations;

        // Edge-Glow: nahe Escape-Grenze leicht boosten, monoton halten
        float edge = clamp01(1.0f - 0.75f * l2);             // 0..1
        x = clamp01(x + 0.15f * edge * (1.0f - x));
        return gtPalette_u8(x, false, t);
    } else {
        float x = clamp01((float)it / (float)maxIterations);
        return gtPalette_u8(x, false, t);
    }
}

// ============================================================================
// Direkter Mandelbrot-Kernel (ohne Referenz-Orbit / Perturbation)
// ============================================================================
__global__ __launch_bounds__(256)
void mandelbrotKernel(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    int w, int h, float zoom, float2 center, int maxIter, float tSec)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx = y * w + x;

    // Mapping via Projektfunktion (konsistent, keine Verzerrung)
    const float2 c = pixelToComplex(
        (double)x + 0.5, (double)y + 0.5,
        w, h,
        (double)center.x, (double)center.y,
        (double)zoom
    );

    // Early interior exit
    if (insideMainCardioidOrBulb(c.x, c.y)){
        out[idx]     = make_uchar4(10,12,16,255);
        iterOut[idx] = maxIter;   // innen = maxIter (Heatmap-Vertrag)
        return;
    }

    // Direkte Iteration z_{n+1} = z_n^2 + c
    float zx = 0.0f, zy = 0.0f;
    int   it = maxIter;           // default: gilt als "innen"
    const float esc2 = 4.0f;

    #pragma unroll 1
    for (int i=0; i<maxIter; ++i){
        const float x2 = zx*zx, y2 = zy*zy;

        // Escape testen vor dem Update
        if (x2 + y2 > esc2){
            it = i;               // Iteration der Flucht
            break;
        }

        const float xt = x2 - y2 + c.x;
        zy = __fmaf_rn(2.0f*zx, zy, c.y);
        zx = xt;
    }

    out[idx]     = gtColor_fromSmoothState(it, maxIter, zx, zy, tSec);
    iterOut[idx] = it;
}

// ============================================================================
// Öffentliche API (wie im funktionierenden Build, Call-Sites unverändert)
// ============================================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;
    static clk::time_point t0;
    static bool t0_init = false;
    if (!t0_init){ t0 = clk::now(); t0_init = true; }
    const float tSec = (float)std::chrono::duration<double>(clk::now() - t0).count();

    if (!out || !d_it || w <= 0 || h <= 0 || maxIter <= 0){
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",
                       (void*)out, (void*)d_it, w, h, maxIter);
        return;
    }

    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    mandelbrotKernel<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter, tSec);

    if constexpr (Settings::performanceLogging){
        cudaDeviceSynchronize();
        const double ms = 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull direct+anim kern=%.2f ms itMax=%d", ms, maxIter);
    }
}
