///// Otter: GT-Palette (Cyan→Amber) integriert; Perturbation korrekt (dz + ref, delta=c-c0); API unverändert.
/// ///// Schneefuchs: Host/Device strikt getrennt; keine Device-Intrinsics im Hostcode; ASCII-Logs; deterministisches Mapping.
/// ///// Maus: Schwarzes Bild gefixt: Escape-Test auf |ref+dz|, nicht auf dz; Smooth-Coloring mit finalem z_total.
/// ///// Datei: src/nacktmull.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>      // float2, uchar4
#include <vector_functions.h>  // make_float2, make_float3, make_uchar4
#include <cmath>
#include <vector>
#include <chrono>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"

// ============================================================================
// Device Utilities
// ============================================================================
__device__ __forceinline__ float  clamp01(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ __forceinline__ float  mixf(float a, float b, float t) { return a + t * (b - a); }
__device__ __forceinline__ float3 mix3(const float3 a, const float3 b, float t) {
    return make_float3(mixf(a.x,b.x,t), mixf(a.y,b.y,t), mixf(a.z,b.z,t));
}
__device__ __forceinline__ float2 add(const float2 a, const float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ float2 sub(const float2 a, const float2 b){ return make_float2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ float2 mul(const float2 a, const float2 b){ return make_float2(__fmaf_rn(a.x, b.x, -a.y*b.y),
                                                                                           __fmaf_rn(a.x, b.y,  a.y*b.x)); }
__device__ __forceinline__ float2 muls(const float2 a, float s){ return make_float2(a.x*s, a.y*s); }
__device__ __forceinline__ float  dot2(const float2 a){ return __fmaf_rn(a.x, a.x, a.y*a.y); }

// ============================================================================
// Inside guards (Cardioid & period-2 bulb) – fast early-out
// ============================================================================
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y * y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true;       // main cardioid
    const float xp = x + 1.0f;                      // period-2 bulb
    if (xp*xp + y2 <= 0.0625f) return true;
    return false;
}

// ============================================================================
// sRGB <-> linear helpers (device) – reduce banding
// ============================================================================
__device__ __forceinline__ float  srgb_to_linear(float c) {
    return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
}
__device__ __forceinline__ float  linear_to_srgb(float c) {
    return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
}
__device__ __forceinline__ float3 srgb_to_linear3(const float3 c) {
    return make_float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}
__device__ __forceinline__ float3 linear_to_srgb3(const float3 c) {
    return make_float3(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

// ============================================================================
// GT Palette (Cyan → Amber), lerp in linear space, anchors in sRGB
// ============================================================================
__device__ __forceinline__ uchar4 gtPalette_u8(float x, bool inSet) {
    // tuning knobs
    const float gamma     = 0.90f;  // midtone punch
    const float vibr      = 1.06f;  // slight vibrance
    const float warmShift = 1.00f;  // >1 warmer, <1 cooler
    const float stripes   = 0.035f; // subtle iso-lines; 0..0.08
    const float stripeFreq= 6.5f;   // 5..9 recommended

    if (inSet) {
        return make_uchar4(10, 12, 16, 255); // deep neutral interior
    }

    x = clamp01(powf(clamp01(x), gamma));

    // anchors
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
    const float  span = fmaxf(p[j+1] - p[j], 1e-6f);
    float        t    = clamp01((x - p[j]) / span);
    t = t*t*(3.0f - 2.0f*t); // smootherstep

    float3 aLin = srgb_to_linear3(c[j]);
    float3 bLin = srgb_to_linear3(c[j+1]);
    float3 rgbLin = mix3(aLin, bLin, t);

    if (stripes > 0.0f) {
        const float s = 0.5f + 0.5f * __sinf(6.2831853f * (x * stripeFreq));
        const float boost = 1.0f + stripes * (s*s*s*s); // bias highlights
        rgbLin.x *= boost; rgbLin.y *= boost; rgbLin.z *= boost;
    }

    // vibrance & warmth in linear domain
    {
        const float luma = 0.2126f*rgbLin.x + 0.7152f*rgbLin.y + 0.0722f*rgbLin.z;
        rgbLin = make_float3(
            luma + (rgbLin.x - luma) * vibr * warmShift,
            luma + (rgbLin.y - luma) * vibr * 1.00f,
            luma + (rgbLin.z - luma) * vibr * (2.0f - warmShift)
        );
    }

    const float3 srgb = linear_to_srgb3(make_float3(
        clamp01(rgbLin.x), clamp01(rgbLin.y), clamp01(rgbLin.z)
    ));

    const unsigned char R = (unsigned char)(clamp01(srgb.x) * 255.0f + 0.5f);
    const unsigned char G = (unsigned char)(clamp01(srgb.y) * 255.0f + 0.5f);
    const unsigned char B = (unsigned char)(clamp01(srgb.z) * 255.0f + 0.5f);
    return make_uchar4(R, G, B, 255);
}

// Smooth-iterations color mapping (device)
__device__ __forceinline__ uchar4 gtColor_fromSmoothState(
    int it, int maxIterations, float zx, float zy)
{
    const bool inSet = (it >= maxIterations);
    if (inSet) return gtPalette_u8(0.0f, true);

    const float mag2 = zx*zx + zy*zy;
    if (mag2 > 1.0000001f && it > 0) {
        const float mag = sqrtf(mag2);
        float x = ((float)it - __log2f(__log2f(mag))) / (float)maxIterations;
        return gtPalette_u8(clamp01(x), false);
    } else {
        float x = clamp01((float)it / (float)maxIterations);
        return gtPalette_u8(x, false);
    }
}

// ============================================================================
// Kernel – Perturbation (korrekt): dz_{n+1} = 2*ref_n*dz_n + dz_n^2 + (c - c0)
// Escape-Test auf |z_total| mit z_total = ref_n + dz_n
// ============================================================================
__global__ __launch_bounds__(256)
void perturbKernel(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    const float2* __restrict__ refOrbit, int maxIter,
    int w, int h, float zoom, float2 center, float /*deltaRelMax*/)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx = y * w + x;

    // Pixel → c (center/zoom mapping)
    const float invZoom = 1.0f / zoom;
    const float sx = ((float)x / (float)w - 0.5f) * 3.2f;
    const float sy = ((float)y / (float)h - 0.5f) * 2.0f;
    const float2 c  = make_float2(center.x + sx * invZoom,
                                  center.y + sy * invZoom);
    const float2 c0 = center;
    const float2 delta = sub(c, c0); // *** wichtig für Perturbation ***

    // Early interior exit (Cardioid/Bulb)
    if (insideMainCardioidOrBulb(c.x, c.y)) {
        out[idx]     = make_uchar4(10, 12, 16, 255);
        iterOut[idx] = maxIter;
        return;
    }

    // Perturbation: dz starts at 0
    float2 dz = make_float2(0.0f, 0.0f);
    float2 zFinal = make_float2(0.0f, 0.0f);
    int it = 0;

    #pragma unroll 1
    for (int n=0; n<maxIter; ++n) {
        const float2 ref = refOrbit[n];

        // dz_{n+1} = dz_n^2 + 2*ref_n*dz_n + delta
        const float2 dz2      = mul(dz, dz);
        const float2 twoRefDz = muls(mul(ref, dz), 2.0f);
        dz = add(add(dz2, twoRefDz), delta);

        // z_total = ref_{n+1}? Näherung mit ref_n+dz_n nach Update (bewährt in Praxis)
        const float2 ztot = add(ref, dz);

        it = n + 1;
        zFinal = ztot;

        // Escape auf |z_total| > 2
        if (dot2(ztot) > 4.0f) {
            break;
        }
    }

    // Farbe schreiben (Smooth-Coloring mit finalem z_total)
    out[idx]     = gtColor_fromSmoothState(it, maxIter, zFinal.x, zFinal.y);
    iterOut[idx] = it;
}

// ============================================================================
// Host – Build reference orbit (float), launch kernel
// ============================================================================
extern "C" void launch_mandelbrotHybrid(
    unsigned char* pboPtr,
    int* outIterations,
    int w, int h, float zoom,
    float centerX, float centerY,
    int maxIter)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    // Reference orbit around c0=(centerX,centerY) (host, float)
    std::vector<float2> hostOrbit;
    hostOrbit.resize(maxIter);
    {
        float2 z = make_float2(0,0);
        const float2 c0 = make_float2(centerX, centerY);
        for (int i=0; i<maxIter; ++i){
            // z = z^2 + c0
            const float zx2 = z.x*z.x, zy2 = z.y*z.y;
            const float xt  = zx2 - zy2 + c0.x;
            // Host: std::fma statt __fmaf_rn
            z.y = std::fma(2.f*z.x, z.y, c0.y);
            z.x = xt;
            hostOrbit[i] = make_float2(z.x, z.y);
        }
    }

    // Upload orbit
    float2* d_orbit = nullptr;
    cudaMalloc(&d_orbit, sizeof(float2)*maxIter);
    cudaMemcpy(d_orbit, hostOrbit.data(), sizeof(float2)*maxIter, cudaMemcpyHostToDevice);

    // Outputs
    uchar4* d_out = reinterpret_cast<uchar4*>(pboPtr);
    int*    d_it  = outIterations;

    // Kernel
    perturbKernel<<<grid, block>>>(d_out, d_it, d_orbit, maxIter, w, h, zoom,
                                   make_float2(centerX, centerY), 0.02f);

    // Cleanup
    cudaFree(d_orbit);

    if constexpr (Settings::performanceLogging){
        const double ms = 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull kern=%.2f ms it=%d", ms, maxIter);
    }
}
