///// Otter: Keks 3 – Periodizitäts-Probe via Flag; Keks-0 ms-Timing bleibt; kein Default-Verhaltenswechsel.
///// Schneefuchs: Check alle N Iter.; EPS² aus Settings; Treffer ⇒ it=maxIter (bounded), escaped=false.
///  Maus: ASCII-Logs, deterministisch; if constexpr-Gate eliminiert Overhead, wenn disabled.
///// Datei: src/nacktmull.cu
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
__device__ __forceinline__ float clamp01(float x){ return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ __forceinline__ float mixf(float a, float b, float t){ return a + t*(b - a); }
__device__ __forceinline__ float3 mix3(const float3 a, const float3 b, float t){
    return make_float3(mixf(a.x,b.x,t), mixf(a.y,b.y,t), mixf(a.z,b.z,t));
}
__device__ __forceinline__ float smootherstep(float a, float b, float x){
    float t = clamp01((x - a) / fmaxf(b - a, 1e-6f));
    return t*t*(3.0f - 2.0f*t);
}

// sRGB <-> Linear
__device__ __forceinline__ float  srgb_to_linear(float c){
    return (c <= 0.04045f) ? (c/12.92f) : powf((c + 0.055f)/1.055f, 2.4f);
}
__device__ __forceinline__ float  linear_to_srgb(float c){
    return (c <= 0.0031308f) ? (12.92f*c) : (1.055f*powf(c, 1.0f/2.4f) - 0.055f);
}
__device__ __forceinline__ float3 srgb_to_linear3(const float3 c){
    return make_float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}
__device__ __forceinline__ float3 linear_to_srgb3(const float3 c){
    return make_float3(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

// Cardioid / Period-2-Bulb (Early-Out) – Routine vorhanden (Keks 2 übersprungen)
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y*y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true; // main cardioid
    const float xp = x + 1.0f;                // period-2 bulb
    if (xp*xp + y2 <= 0.0625f) return true;
    return false;
}

// ============================================================================
// Hintergrund (linear) – dezenter Verlauf + leichte "Caustics"
// ============================================================================
__device__ __forceinline__ float3 background_linear(float u, float v, float t){
    const float3 A = srgb_to_linear3(make_float3( 8/255.f, 10/255.f, 16/255.f));
    const float3 B = srgb_to_linear3(make_float3(28/255.f, 30/255.f, 44/255.f));
    float w = clamp01(0.25f + 0.75f*(0.6f*u + 0.4f*(1.0f - v)));
    float3 base = mix3(A, B, w);

    // sehr subtil,  ±~1.5 %
    float phase  = 3.0f*u + 2.0f*v + 0.20f*t;
    float phase2 = 5.0f*u - 3.0f*v - 0.15f*t;
    float caust  = 0.5f + 0.5f*__sinf(phase) * (0.5f + 0.5f*__cosf(phase2));
    float boost  = 1.0f + 0.02f*(caust*caust - 0.25f);
    base.x *= boost; base.y *= boost; base.z *= boost;
    return make_float3(clamp01(base.x), clamp01(base.y), clamp01(base.z));
}

// ============================================================================
// GT-Palette (breiter Bogen), linear gemischt + sanfte "Breathing"-Farbbewegung
//  -> Rückgabe in sRGB (0..1)
// ============================================================================
__device__ __forceinline__ float3 gtPalette_srgb(float x, bool inSet, float t){
    const float gamma        = 0.84f;  // mehr Mittentöne
    const float lift         = 0.08f;  // dunkel leicht anheben
    const float baseVibr     = 1.05f;
    const float addVibrMax   = 0.06f;  // zusätzl. Sättigung ab x≈0.1
    const float warmDriftAmp = 0.06f;  // ±6 % Warmverschiebung
    const float warmShift    = 1.00f + warmDriftAmp*__sinf(0.30f*t);
    const float breathAmp    = 0.08f;  // ±0.08 Input-Breathing

    if (inSet){
        // sehr dunkles, leicht bläuliches Glas statt Vollschwarz
        return make_float3(12/255.f, 14/255.f, 20/255.f);
    }

    x = clamp01(powf(clamp01(x), gamma));
    x = clamp01((x + lift) / (1.0f + lift));
    const float xprime = clamp01(x + breathAmp*__sinf(0.80f*t)*x*(1.0f - x));

    // Indigo → Cyan → Mint → Sand → Amber → Rose → Off-White
    const float  p[8] = { 0.00f, 0.10f, 0.22f, 0.38f, 0.55f, 0.72f, 0.88f, 1.00f };
    const float3 c[8] = {
        make_float3(11/255.f, 14/255.f, 26/255.f),
        make_float3(26/255.f, 43/255.f,111/255.f),
        make_float3(30/255.f,166/255.f,209/255.f),
        make_float3(123/255.f,228/255.f,195/255.f),
        make_float3(255/255.f,224/255.f,138/255.f),
        make_float3(247/255.f,165/255.f, 58/255.f),
        make_float3(200/255.f, 72/255.f,122/255.f),
        make_float3(250/255.f,249/255.f,246/255.f)
    };

    int j = 0;
    #pragma unroll
    for (int i=0;i<7;++i){ if (xprime >= p[i]) j = i; }
    const float span = fmaxf(p[j+1]-p[j], 1e-6f);
    float tseg = clamp01((xprime - p[j]) / span);
    tseg = tseg*tseg*(3.0f - 2.0f*tseg);

    float3 aLn = srgb_to_linear3(c[j]);
    float3 bLn = srgb_to_linear3(c[j+1]);
    float3 rgbLn = mix3(aLn, bLn, tseg);

    // Sättigung & Wärme im Linearraum
    const float luma = 0.2126f*rgbLn.x + 0.7152f*rgbLn.y + 0.0722f*rgbLn.z;
    const float vibr = baseVibr + addVibrMax*clamp01((xprime - 0.10f)*(1.0f/0.40f));
    rgbLn = make_float3(
        luma + (rgbLn.x - luma)*vibr*warmShift,
        luma + (rgbLn.y - luma)*vibr*1.00f,
        luma + (rgbLn.z - luma)*vibr*(2.0f - warmShift)
    );

    float3 srgb = linear_to_srgb3(make_float3(clamp01(rgbLn.x), clamp01(rgbLn.y), clamp01(rgbLn.z)));
    return srgb;
}

// ============================================================================
// Farbe (Smooth) + Alpha (Transparenz: DE + Fresnel + Breathing)
// ============================================================================
__device__ __forceinline__ void shade_color_alpha(
    int it, int maxIter,
    float zx, float zy,         // z beim Escape/Ende
    float dx, float dy,         // z' beim Escape/Ende
    bool escaped,               // true, wenn Escape erkannt
    float t,                    // Zeit in Sekunden
    float3& out_srgb, float& out_alpha)
{
    // ---- Farbe ----
    if (it >= maxIter){
        out_srgb = gtPalette_srgb(0.0f, true, t);
    } else {
        const float r2 = zx*zx + zy*zy;
        if (r2 > 1.0000001f && it > 0){
            const float r  = sqrtf(r2);
            const float l2 = __log2f(__log2f(r));
            float x = ((float)it - l2) / (float)maxIter;
            // sanfter Kanten-Punch (monoton, ringfrei)
            float edge = clamp01(1.0f - 0.75f*l2);
            x = clamp01(x + 0.15f*edge*(1.0f - x));
            out_srgb = gtPalette_srgb(x, false, t);
        } else {
            float x = clamp01((float)it / (float)maxIter);
            out_srgb = gtPalette_srgb(x, false, t);
        }
    }

    // ---- Alpha ----
    if (!escaped){
        // Innen: getöntes Glas (leicht durchsichtig)
        out_alpha = 0.18f;
        return;
    }

    // Distance-Estimator: DE ≈ |z|*log(|z|)/|z'|
    const float r     = fmaxf(1e-7f, sqrtf(zx*zx + zy*zy));
    const float dmag  = fmaxf(1e-12f, sqrtf(dx*dx + dy*dy));
    const float DE    = (r * logf(r)) / dmag;

    // Alpha aus 1/DE, nahe Kante dicht, draußen transparent
    const float invDE = 1.0f / (DE + 1e-6f);
    float A = smootherstep(0.02f, 0.20f, invDE);

    // Fresnel-Saum (nahe Escape-Grenze betonen)
    const float l2 = __log2f(__log2f(r));
    const float F  = clamp01(1.0f - 0.80f*l2);
    A = clamp01(A * (0.60f + 0.40f*F));

    // Breathing in (1-A): Transparenz „atmet“ dezent, Kante bleibt stabil
    const float breathe = 0.12f * __sinf(0.60f * t);
    float transp = clamp01(1.0f - A);
    transp = clamp01(transp * (1.0f + breathe));
    out_alpha = clamp01(1.0f - transp);
}

// ============================================================================
// Direkter Mandelbrot-Kernel (Iteration + Ableitung) + optional Periodizität (Keks 3)
// ============================================================================
__global__ __launch_bounds__(256)
void mandelbrotKernel(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    int w, int h, float zoom, float2 center, int maxIter, float tSec)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx = y*w + x;

    const float2 c = pixelToComplex(
        (double)x + 0.5, (double)y + 0.5,
        w, h,
        (double)center.x, (double)center.y,
        (double)zoom
    );

    // Early interior exit (Cardioid/Bulb) – vorhanden
    if (insideMainCardioidOrBulb(c.x, c.y)){
        const float u = ((float)x + 0.5f) / (float)w;
        const float v = ((float)y + 0.5f) / (float)h;
        const float3 bgLn   = background_linear(u, v, tSec);
        const float3 glassS = gtPalette_srgb(0.0f, true, tSec);
        const float3 glassL = srgb_to_linear3(glassS);
        const float  A      = 0.18f; // leicht getöntes Glas
        float3 compLn = make_float3(
            A*glassL.x + (1.0f - A)*bgLn.x,
            A*glassL.y + (1.0f - A)*bgLn.y,
            A*glassL.z + (1.0f - A)*bgLn.z
        );
        const float3 compS = linear_to_srgb3(compLn);
        out[idx] = make_uchar4(
            (unsigned char)(255.0f*clamp01(compS.x) + 0.5f),
            (unsigned char)(255.0f*clamp01(compS.y) + 0.5f),
            (unsigned char)(255.0f*clamp01(compS.z) + 0.5f),
            255
        );
        iterOut[idx] = maxIter; // Heatmap-Vertrag
        return;
    }

    // Iteration z_{n+1} = z_n^2 + c, Ableitung z'_{n+1} = 2*z_n*z'_n + 1
    float zx = 0.0f, zy = 0.0f;   // z
    float dx = 0.0f, dy = 0.0f;   // z'
    int   it = maxIter;
    bool  escaped = false;
    const float esc2 = 4.0f;

    // -------- Keks 3: Periodizitäts-Probe (optional via constexpr-Flag) --------
    float px = 0.0f, py = 0.0f;   // gespeicherte Probe von z
    int   lastProbe = 0;
    const int   perN  = Settings::periodicityCheckInterval; // z. B. 64
    const float eps2  = (float)Settings::periodicityEps2;   // z. B. 1e-14

    #pragma unroll 1
    for (int i=0; i<maxIter; ++i){
        const float x2 = zx*zx, y2 = zy*zy;

        // Escape vor Update → r > 2 garantiert im Escape-Fall
        if (x2 + y2 > esc2){
            it = i;
            escaped = true;
            break;
        }

        // Ableitung (benötigt altes z)
        const float twx = 2.0f*zx;
        const float twy = 2.0f*zy;
        const float ndx = twx*dx - twy*dy + 1.0f;
        const float ndy = twx*dy + twy*dx;

        // z-Update
        const float xt = x2 - y2 + c.x;
        zy = __fmaf_rn(2.0f*zx, zy, c.y);
        zx = xt;

        dx = ndx; dy = ndy;

        // Periodizitäts-Check: nur wenn aktiviert, compile-time entfernt wenn false
        if constexpr (Settings::periodicityEnabled){
            const int step = i - lastProbe;
            if (step >= perN){
                const float dxp = zx - px;
                const float dyp = zy - py;
                const float d2  = dxp*dxp + dyp*dyp;
                if (d2 <= eps2){
                    // (nahe) periodisch → bounded: als „innen“ behandeln
                    it = maxIter;
                    escaped = false;
                    break;
                }
                px = zx; py = zy;
                lastProbe = i;
            }
        }
    }

    // Farbe + Alpha
    float3 frag_srgb; float alpha;
    shade_color_alpha(it, maxIter, zx, zy, dx, dy, escaped, tSec, frag_srgb, alpha);

    // Hintergrund + Komposition (linear)
    const float u = ((float)x + 0.5f) / (float)w;
    const float v = ((float)y + 0.5f) / (float)h;
    const float3 bgLn = background_linear(u, v, tSec);
    const float3 fgLn = srgb_to_linear3(frag_srgb);
    float3 compLn = make_float3(
        alpha*fgLn.x + (1.0f - alpha)*bgLn.x,
        alpha*fgLn.y + (1.0f - alpha)*bgLn.y,
        alpha*fgLn.z + (1.0f - alpha)*bgLn.z
    );
    const float3 compS = linear_to_srgb3(compLn);

    out[idx] = make_uchar4(
        (unsigned char)(255.0f*clamp01(compS.x) + 0.5f),
        (unsigned char)(255.0f*clamp01(compS.y) + 0.5f),
        (unsigned char)(255.0f*clamp01(compS.z) + 0.5f),
        255
    );
    iterOut[idx] = it;
}

// ============================================================================
// Öffentliche API (Signatur unverändert) – Keks-0: korrektes ms-Timing
// ============================================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;
    static clk::time_point anim0;
    static bool anim_init = false;
    if (!anim_init){ anim0 = clk::now(); anim_init = true; }
    const float tSec = (float)std::chrono::duration<double>(clk::now() - anim0).count();
    const auto tStart = clk::now();

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
        const double ms = 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - tStart).count();
        LUCHS_LOG_HOST("[PERF] nacktmull direct+transp kern=%.2f ms itMax=%d", ms, maxIter);
    }

    if constexpr (Settings::debugLogging){
        LUCHS_LOG_HOST("[INFO] periodicity enabled=%d N=%d eps2=%.3e",
                       (int)Settings::periodicityEnabled,
                       (int)Settings::periodicityCheckInterval,
                       (double)Settings::periodicityEps2);
    }
}
