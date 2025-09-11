///// Otter: Direkte Mandelbrot-Iteration (ohne Referenz-Orbit), GT-Palette (Cyan→Amber), Smooth-Coloring.
///  Schneefuchs: API & Mapping unverändert (pixelToComplex), deterministisch, ASCII-Logs.
///  Maus: Ringe entfernt (Stripes=0.0), Heatmap-Vertrag unverändert (innen = maxIter).

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
// GT-Palette (Cyan→Amber), Interpolation im Linearraum
// ============================================================================
__device__ __forceinline__ uchar4 gtPalette_u8(float x, bool inSet){
    // Tuning – Ringe aus (stripes = 0.0f)
    const float gamma      = 0.90f;
    const float vibr       = 1.06f;
    const float warmShift  = 1.00f;
    const float stripes    = 0.0f;  // dekorative Isolinien deaktiviert
    const float stripeFreq = 6.5f;

    if (inSet) return make_uchar4(10, 12, 16, 255);

    x = clamp01(powf(clamp01(x), gamma));

    const float  p[8] = { 0.00f, 0.12f, 0.25f, 0.42f, 0.60f, 0.78f, 0.95f, 1.00f };
    const float3 c[8] = {
        make_float3( 8/255.f,  9/255.f, 15/255.f), // #08090F
        make_float3(17/255.f, 45/255.f, 95/255.f), // #112D5F
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
    float t = clamp01((x - p[j]) / span);
    t = t*t*(3.0f - 2.0f*t); // smootherstep

    float3 aLin  = srgb_to_linear3(c[j]);
    float3 bLin  = srgb_to_linear3(c[j+1]);
    float3 rgbLn = mix3(aLin, bLin, t);

    if (stripes > 0.0f){
        const float s   = 0.5f + 0.5f * __sinf(6.2831853f * (x * stripeFreq));
        const float sat = 1.0f + stripes * (s*s*s*s); // Sättigungsboost (Luma-neutral)
        const float L   = 0.2126f*rgbLn.x + 0.7152f*rgbLn.y + 0.0722f*rgbLn.z;
        rgbLn.x = L + (rgbLn.x - L)*sat;
        rgbLn.y = L + (rgbLn.y - L)*sat;
        rgbLn.z = L + (rgbLn.z - L)*sat;
    }

    // leichte Vibrance/Warmth im Linearraum
    {
        const float luma = 0.2126f*rgbLn.x + 0.7152f*rgbLn.y + 0.0722f*rgbLn.z;
        rgbLn = make_float3(
            luma + (rgbLn.x - luma) * vibr * warmShift,
            luma + (rgbLn.y - luma) * vibr * 1.00f,
            luma + (rgbLn.z - luma) * vibr * (2.0f - warmShift)
        );
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

// Smooth-Iterations → Farbwert
__device__ __forceinline__ uchar4 gtColor_fromSmoothState(int it, int maxIterations, float zx, float zy){
    const bool inSet = (it >= maxIterations);
    if (inSet) return gtPalette_u8(0.0f, true);

    const float mag2 = zx*zx + zy*zy;
    if (mag2 > 1.0000001f && it > 0){
        const float mag = sqrtf(mag2);
        float x = ((float)it - __log2f(__log2f(mag))) / (float)maxIterations;
        return gtPalette_u8(clamp01(x), false);
    } else {
        float x = clamp01((float)it / (float)maxIterations);
        return gtPalette_u8(x, false);
    }
}

// ============================================================================
// Direkter Mandelbrot-Kernel (ohne Referenz-Orbit / Perturbation)
// ============================================================================
__global__ __launch_bounds__(256)
void mandelbrotKernel(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    int w, int h, float zoom, float2 center, int maxIter)
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

        // Escape testen wie im stabilen Build: vor dem Update
        if (x2 + y2 > esc2){
            it = i;               // Iteration, in der Escape festgestellt wurde
            break;
        }

        const float xt = x2 - y2 + c.x;
        zy = __fmaf_rn(2.0f*zx, zy, c.y);
        zx = xt;
    }

    out[idx]     = gtColor_fromSmoothState(it, maxIter, zx, zy);
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
    const auto t0 = clk::now();

    if (!out || !d_it || w <= 0 || h <= 0 || maxIter <= 0){
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",
                       (void*)out, (void*)d_it, w, h, maxIter);
        return;
    }

    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    mandelbrotKernel<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter);

    if constexpr (Settings::performanceLogging){
        cudaDeviceSynchronize();
        const double ms = 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull direct kern=%.2f ms itMax=%d", ms, maxIter);
    }
}
