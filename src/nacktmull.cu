///// Otter: GT-Palette (Cyan→Amber) integriert; Perturbation korrekt (dz + ref, delta=c-c0); API unverändert.
/// ///// Schneefuchs: Host/Device strikt getrennt; keine Device-Intrinsics im Hostcode; ASCII-Logs; deterministisches Mapping.
/// ///// Maus: Heatmap/Zoom-Vertrag wiederhergestellt: kein internes maxIter-Clamping; Guard+Fallback wie im stabilen Build.
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
#include "nacktmull_math.cuh"  // pixelToComplex(...)

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
__device__ __forceinline__ float2 mul(const float2 a, const float2 b){
    return make_float2(__fmaf_rn(a.x, b.x, -a.y*b.y),
                       __fmaf_rn(a.x, b.y,  a.y*b.x));
}
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
// Host: Double-Double für Referenz-Orbit (stabil wie im funktionierenden Build)
// ============================================================================
namespace dd {
    struct dd { double hi, lo; };

    inline dd make(double x){ return {x, 0.0}; }

    inline dd two_sum(double a, double b){
        double s = a + b;
        double bb = s - a;
        double e = (a - (s - bb)) + (b - bb);
        return {s, e};
    }
    inline dd quick_two_sum(double a, double b){
        double s = a + b;
        double e = b - (s - a);
        return {s, e};
    }
    inline dd add(dd a, dd b){
        dd s = two_sum(a.hi, b.hi);
        double t = a.lo + b.lo + s.lo;
        return quick_two_sum(s.hi, t);
    }
    inline dd mul(dd a, dd b){
        double p = a.hi * b.hi;
        double e = std::fma(a.hi, b.hi, -p);
        e += a.hi * b.lo + a.lo * b.hi;
        return quick_two_sum(p, e);
    }
}

static void buildReferenceOrbitDD(std::vector<float2>& out, int maxIter, double cref_x, double cref_y){
    out.resize((size_t)maxIter);
    dd::dd zx = dd::make(0.0), zy = dd::make(0.0);
    dd::dd cr = dd::make(cref_x), ci = dd::make(cref_y);

    for(int i=0;i<maxIter;i++){
        dd::dd x2 = dd::mul(zx, zx);
        dd::dd y2 = dd::mul(zy, zy);
        dd::dd xy = dd::mul(zx, zy);
        dd::dd zr = dd::add(dd::add(x2, dd::make(-y2.hi)), cr);      // (x^2 - y^2) + cr
        dd::dd zi = dd::add(dd::make(2.0 * xy.hi), ci);               // 2*x*y + ci
        zx = zr; zy = zi;
        out[(size_t)i] = make_float2((float)zx.hi, (float)zy.hi);
    }
}

// ============================================================================
// Kernel – Perturbation (mit Guard+Fallback): dz_{n+1} = dz^2 + 2*ref*dz + (c - c0)
// Escape-Test auf |z_total| mit z_total = ref + dz
// ============================================================================
__global__ __launch_bounds__(256)
void perturbKernel(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    const float2* __restrict__ refOrbit, int maxIter,
    int w, int h, float zoom, float2 center, float deltaRelMax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx = y * w + x;

    // Mapping (projektweit konsistent)
    const float2 c = pixelToComplex(
        (double)x + 0.5, (double)y + 0.5,
        w, h,
        (double)center.x, (double)center.y,
        (double)zoom
    );
    const float2 c0    = make_float2(center.x, center.y);
    const float2 delta = sub(c, c0); // Δc = c - c0

    // Early interior exit (Cardioid/Bulb)
    if (insideMainCardioidOrBulb(c.x, c.y)) {
        out[idx]     = make_uchar4(10, 12, 16, 255);
        iterOut[idx] = maxIter;   // Innen wie im funktionierenden Build
        return;
    }

    float2 dz = make_float2(0.0f, 0.0f);
    float2 zFinal = make_float2(0.0f, 0.0f);
    int it = 0;

    const float esc2 = 4.0f;
    const float tau2 = deltaRelMax * deltaRelMax; // Guard-Schwelle (relativ zu |ref|)

    #pragma unroll 1
    for (int n=0; n<maxIter; ++n) {
        const float2 ref = refOrbit[n];

        // dz_{n+1} = dz^2 + 2*ref*dz + delta
        const float2 dz2      = mul(dz, dz);
        const float2 twoRefDz = muls(mul(ref, dz), 2.0f);
        dz = add(add(dz2, twoRefDz), delta);

        // z_total ~ ref + dz (nach Update)
        const float2 ztot = add(ref, dz);

        it = n + 1;
        zFinal = ztot;

        // Escape auf |z_total| > 2
        if (dot2(ztot) > esc2) { goto ESCAPE; }

        // Stabilitäts-Guard → Fallback auf direkte Iteration ab ztot
        if (dot2(dz) > tau2 * fmaxf(dot2(ref), 1e-24f)) {
            float zx = ztot.x, zy = ztot.y;
            for (int m=n+1; m<maxIter; ++m){
                const float x2 = zx*zx, y2 = zy*zy;
                if (x2 + y2 > 4.f){ it = m; zFinal = make_float2(zx,zy); goto ESCAPE; }
                const float xt = x2 - y2 + c.x;
                zy = __fmaf_rn(2.f*zx, zy, c.y);
                zx = xt;
            }
            it = maxIter; zFinal = make_float2(zx,zy); goto ESCAPE;
        }
    }
    it = maxIter; // nicht entkommen
ESCAPE:
    {
        out[idx]     = gtColor_fromSmoothState(it, maxIter, zFinal.x, zFinal.y);
        iterOut[idx] = it;
    }
}

// ============================================================================
// Host – Referenz-Orbit (DD), Launch
// ============================================================================
// *** WICHTIG: API exakt wie im funktionierenden Build ***
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();

    if (!out || !d_it || w <= 0 || h <= 0 || maxIter <= 0) {
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",
                       (void*)out, (void*)d_it, w, h, maxIter);
        return;
    }

    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    // Referenz-Orbit um c0=(offset.x, offset.y) (Host, Double-Double → float2)
    std::vector<float2> hostOrbit;
    hostOrbit.reserve((size_t)maxIter);
    buildReferenceOrbitDD(hostOrbit, maxIter, (double)offset.x, (double)offset.y);

    // Upload orbit
    float2* d_orbit = nullptr;
    cudaMalloc(&d_orbit, sizeof(float2)*maxIter);
    cudaMemcpy(d_orbit, hostOrbit.data(), sizeof(float2)*maxIter, cudaMemcpyHostToDevice);

    // Kernel
    perturbKernel<<<grid, block>>>(out, d_it, d_orbit, maxIter, w, h, zoom, offset, 0.02f);

    // Cleanup
    cudaFree(d_orbit);

    if constexpr (Settings::performanceLogging){
        const double ms = 1e-3 * (double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull kern=%.2f ms itMax=%d", ms, maxIter);
    }
}
