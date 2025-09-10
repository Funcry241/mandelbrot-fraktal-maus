///// Otter: Perturbation/Series Path – Referenz-Orbit (DD) + GPU-Delta; API unveraendert.
///// Schneefuchs: Deterministisch, ASCII-only; Mapping konsistent (center+zoom); kompakte PERF-Logs.
///// Maus: Keine Overlays/Sprites; kein CUDA_CHECK; fruehe Rueckgaben bei Fehlern.
///// Datei: src/nacktmull.cu
//  CUDA 13 Optimierungen (ohne Verhaltensaenderung):
//   - __launch_bounds__(256) fuer bessere Occupancy (Host nutzt 32x8 Threads).
//   - FMA/Math-Intrinsics im Hotpath (__fmaf_rn, __powf, __expf).
//   - Guard ohne sqrt (vergleich mit quadratischen Normen) → exakt aequivalent.
//   - c_ref nur 1x pro Block berechnet und via Shared Memory verteilt.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>      // float2, uchar4
#include <vector_functions.h>  // make_float2, make_float3, make_uchar4
#include <cmath>
#include <vector>
#include <chrono>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "nacktmull_math.cuh"  // pixelToComplex(...)
#include "common.hpp"          // Projektkonstanz (keine CUDA_CHECK-Nutzung)

// ------------------------------------------------------------
// Cardioid/Period-2-Bulb Test (Host/Device, inline)
// ------------------------------------------------------------
static __host__ __device__ inline bool insideMainCardioidOrBulb(float x, float y){
    float xm = x - 0.25f;
    float q  = xm * xm + y * y;
    if (q * (q + xm) <= 0.25f * y * y) return true; // main cardioid
    float xp = x + 1.0f;
    if (xp * xp + y * y <= 0.0625f)   return true;  // period-2 bulb
    return false;
}

// ------------------------------------------------------------
// Double-Double – minimal (add/mul) fuer Host-Referenz-Orbit
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// Referenz-Orbit: z_{n+1} = z_n^2 + c_ref  (Host, DD)
// Ergebnis: float2 pro Schritt (kompakt), Laenge = maxIter
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// GPU-Kernel: Perturbation (CUDA 13-optimiert, verhaltensgleich)
// δ_{n+1} = 2*z_ref[n]*δ_n + Δc;   z ≈ z_ref[n] + δ
// Escape, wenn |z|^2 > 4.
// Fallback: direkte Iteration, falls |δ| relativ zu |z_ref| zu gross.
// ------------------------------------------------------------
namespace {
    __device__ __forceinline__ float2 f2_add(float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
    __device__ __forceinline__ float2 f2_mul(float2 a, float2 b){
        // (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
        return make_float2(__fmaf_rn(a.x, b.x, -a.y*b.y),
                           __fmaf_rn(a.x, b.y,  a.y*b.x));
    }
    __device__ __forceinline__ float  f2_abs2(float2 a){ return __fmaf_rn(a.x, a.x, a.y*a.y); }

    // --- Neu: GT-Palette (Cyan→Amber) als Drop-in, lerp in Linearraum (sRGB-Anker) ---
    __device__ __forceinline__ float  _srgb_to_linear(float c) {
        return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
    }
    __device__ __forceinline__ float  _linear_to_srgb(float c) {
        return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
    }
    __device__ __forceinline__ float3 _srgb_to_linear3(const float3 c) {
        return make_float3(_srgb_to_linear(c.x), _srgb_to_linear(c.y), _srgb_to_linear(c.z));
    }
    __device__ __forceinline__ float3 _linear_to_srgb3(const float3 c) {
        return make_float3(_linear_to_srgb(c.x), _linear_to_srgb(c.y), _linear_to_srgb(c.z));
    }
    __device__ __forceinline__ float  _clamp01(float x){ return fminf(1.0f, fmaxf(0.0f, x)); }
    __device__ __forceinline__ float  _mixf(float a, float b, float t){ return a + t*(b-a); }
    __device__ __forceinline__ float3 _mix3(float3 a, float3 b, float t){
        return make_float3(_mixf(a.x,b.x,t), _mixf(a.y,b.y,t), _mixf(a.z,b.z,t));
    }

    // Minimal-invasive Ersetzung: gleiche Signatur (float3 0..1), nur Farbraum/Palette neu.
    __device__ __forceinline__ float3 shade_from_iter(int it, int maxIter){
        // vorhandene Helligkeitskurve beibehalten (kompatibel zum bisherigen Look)
        float t = fminf(1.0f, (float)it / (float)maxIter);
        float v = 1.f - __expf(-1.75f * t);
        v = __powf(fmaxf(0.f, v), 0.90f); // "gamma" ~ 0.90

        // GT-Ankerfarben in sRGB
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

        // Segment finden (in v)
        int j = 0;
        #pragma unroll
        for (int i=0;i<7;++i){ if (v >= p[i]) j = i; }
        const float span = fmaxf(p[j+1]-p[j], 1e-6f);
        float tseg = _clamp01((v - p[j]) / span);
        tseg = tseg*tseg*(3.f - 2.f*tseg); // smootherstep

        // sRGB→Linear, lerp, Linear→sRGB
        const float3 aLin = _srgb_to_linear3(c[j]);
        const float3 bLin = _srgb_to_linear3(c[j+1]);
        float3 rgbLin = _mix3(aLin, bLin, tseg);

        // sehr dezente Isolinien (Highlight-betont, banding-arm)
        const float stripes = 0.035f, stripeFreq = 6.5f;
        if (stripes > 0.f){
            const float s = 0.5f + 0.5f * __sinf(6.2831853f * (v * stripeFreq));
            const float boost = 1.0f + stripes * (s*s*s*s);
            rgbLin.x *= boost; rgbLin.y *= boost; rgbLin.z *= boost;
        }

        // leichte Vibrance/Warmshift (Linearraum)
        const float vibr=1.06f, warm=1.00f;
        const float luma = 0.2126f*rgbLin.x + 0.7152f*rgbLin.y + 0.0722f*rgbLin.z;
        rgbLin = make_float3(
            luma + (rgbLin.x - luma) * vibr * warm,
            luma + (rgbLin.y - luma) * vibr * 1.0f,
            luma + (rgbLin.z - luma) * vibr * (2.0f - warm)
        );

        float3 srgb = _linear_to_srgb3(make_float3(
            _clamp01(rgbLin.x), _clamp01(rgbLin.y), _clamp01(rgbLin.z)
        ));
        return make_float3(_clamp01(srgb.x), _clamp01(srgb.y), _clamp01(srgb.z));
    }

    __global__ __launch_bounds__(256)
    void perturbKernel(
        uchar4* __restrict__ out, int* __restrict__ iterOut,
        const float2* __restrict__ refOrbit, int maxIter,
        int w, int h, float zoom, float2 center, float deltaRelMax)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;
        const int idx = y*w + x;

        // c_ref einmal pro Block berechnen (Device), dann teilen → identische Werte, weniger Arbeit.
        __shared__ float2 s_c_ref;
        if (threadIdx.x == 0 && threadIdx.y == 0){
            s_c_ref = pixelToComplex(0.5 + 0.5*w, 0.5 + 0.5*h, w, h,
                                     (double)center.x, (double)center.y, (double)zoom);
        }
        __syncthreads();

        // Konsistentes Mapping pro Pixel (pixelzentriert, projekteigene Funktion)
        const float2 c   = pixelToComplex(x + 0.5, y + 0.5, w, h,
                                          (double)center.x, (double)center.y, (double)zoom);
        const float2 dC  = make_float2(c.x - s_c_ref.x, c.y - s_c_ref.y);

        if (insideMainCardioidOrBulb(c.x, c.y)){
            out[idx] = make_uchar4(0,0,0,255);
            iterOut[idx] = maxIter;
            return;
        }

        float2 delta = make_float2(0.f,0.f);
        int it = 0;
        const float esc2 = 4.0f;

        // Guard ohne sqrt: vergleiche quadratische Normen (aequivalent zur alten Bedingung).
        const float tau2 = deltaRelMax * deltaRelMax;

        for(int n=0; n<maxIter; ++n){
            const float2 zref = refOrbit[n];
            const float2 twoZ = make_float2(2.f*zref.x, 2.f*zref.y);
            delta = f2_add(f2_mul(twoZ, delta), dC);  // δ_{n+1}

            const float2 z = f2_add(zref, delta);
            const float r2 = f2_abs2(z);
            if (r2 > esc2){ it = n; goto ESCAPE; }

            // alt: delMag > deltaRelMax * refMag  (mit sqrt)
            // neu (aequivalent): |δ|^2 > (deltaRelMax^2) * |z_ref|^2
            if (f2_abs2(delta) > tau2 * fmaxf(f2_abs2(zref), 1e-24f)){
                // Fallback: direkte Iteration ab aktuellem Zustand
                float zx = z.x, zy = z.y;
                for(int m=n+1; m<maxIter; ++m){
                    const float x2 = zx*zx, y2 = zy*zy;
                    if (x2 + y2 > 4.f){ it = m; goto ESCAPE; }
                    const float xt = x2 - y2 + c.x;
                    zy = __fmaf_rn(2.f*zx, zy, c.y);
                    zx = xt;
                }
                it = maxIter; goto ESCAPE;
            }
        }
        it = maxIter;
    ESCAPE:
        {
            const float3 col = shade_from_iter(it, maxIter);
            out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                   (unsigned char)(255.f*col.y),
                                   (unsigned char)(255.f*col.z), 255);
            iterOut[idx] = it;
        }
    }
}

// ============================================================
// Oeffentliche API – ersetzt alte Hybrid-Funktion
// Signatur bleibt identisch: launch_mandelbrotHybrid(...)
// Keine Exceptions (extern "C"); eigene Fehlerpruefung + early return.
// ============================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    auto ok = [](cudaError_t e, const char* op)->bool {
        if (e != cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA][ERR] %s failed: %d (%s)", op, int(e), cudaGetErrorString(e));
            return false;
        }
        return true;
    };

    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();

    if (!out || !d_it || w <= 0 || h <= 0 || maxIter <= 0) {
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",
                       (void*)out, (void*)d_it, w, h, maxIter);
        return;
    }

    // 1) Referenzpunkt = Kamera-Center (konsistent zum Mapping)
    const double cref_x = (double)offset.x;
    const double cref_y = (double)offset.y;

    // 2) Referenz-Orbit (Host, DD)
    static std::vector<float2> hRef;
    try {
        buildReferenceOrbitDD(hRef, maxIter, cref_x, cref_y);
    } catch(...) {
        LUCHS_LOG_HOST("[NACKTMULL][ERR] buildReferenceOrbitDD threw; aborting frame");
        return;
    }

    // 3) Device-Puffer (only-grow)
    static float2* dRef = nullptr; static size_t dCap = 0;
    const size_t need = (size_t)maxIter * sizeof(float2);
    if (need > dCap){
        if (dRef && !ok(cudaFree(dRef), "cudaFree(dRef)")) { dRef = nullptr; dCap = 0; return; }
        if (!ok(cudaMalloc(&dRef, need), "cudaMalloc(dRef)")) return;
        dCap = need;
    }
    if (!ok(cudaMemcpy(dRef, hRef.data(), need, cudaMemcpyHostToDevice), "cudaMemcpy H2D(ref)")) return;

    // 4) Kernel
    const dim3 block(32, 8);                  // 256 Threads
    const dim3 grid((w + block.x - 1)/block.x,
                    (h + block.y - 1)/block.y);
    const float deltaRelMax = 1e-3f;          // unveraendert (nur Performance-Optimierung)

    perturbKernel<<<grid, block>>>(out, d_it, dRef, maxIter, w, h, zoom, offset, deltaRelMax);
    if (!ok(cudaGetLastError(), "perturbKernel launch")) return;

    // 5) Perf-Log (optional)
    if (Settings::performanceLogging){
        if (!ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull kern=%.2f ms it=%d", ms, maxIter);
    }
}
