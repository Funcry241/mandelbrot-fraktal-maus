///// Otter: Perturbation/Series Path – Referenz-Orbit (DD) + GPU-Delta; API unveraendert.
///// Schneefuchs: Deterministisch, ASCII-only; Mapping konsistent (center+zoom); kompakte PERF-Logs.
///// Maus: Keine Overlays/Sprites; kein CUDA_CHECK; fruehe Rueckgaben bei Fehlern.
//  Optimiert (CUDA 13):
//   • Wächter ohne sqrt (quadratische Norm) + 2-Step-Hysterese → stabile Geometrie, weniger Umschaltbänder.
//   • Höhere Standardschwelle (tau_on = 5e-3), Vergleich gegen |z| (statt nur |z_ref|).
//   • c_ref einmalig auf Host berechnet und als Kernel-Arg übergeben (spart pro Thread 1x pixelToComplex).
//   • __launch_bounds__(256,2), FMA-Intrinsics, kein unnötiges double im Hotpath.
//   • Performance ≥ vorher (weniger sqrt, seltenerer Fallback, weniger pro-Thread Arbeit).

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
// Referenz-Orbit (Host, DD) -> float2[maxIter]
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
// GPU-Kernel: Perturbation mit robustem Wächter
// δ_{n+1} = 2*z_ref[n]*δ_n + Δc;   z ≈ z_ref[n] + δ
// Escape, wenn |z|^2 > 4.  Fallback nur bei stabiler Überschreitung.
// ------------------------------------------------------------
namespace {
    __device__ __forceinline__ float2 f2_add(float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
    __device__ __forceinline__ float2 f2_mul(float2 a, float2 b){ return make_float2(a.x*b.x - a.y*b.y, __fmaf_rn(a.x, b.y, a.y*b.x)); }
    __device__ __forceinline__ float  f2_abs2(float2 a){ return __fmaf_rn(a.x, a.x, a.y*a.y); }

    __device__ inline float3 shade_from_iter(int it, int maxIter){
        float t = fminf(1.0f, (float)it / (float)maxIter);
        float v = 1.f - __expf(-1.75f * t);
        v = __powf(fmaxf(0.f, v), 0.80f);
        v = v + 0.08f * (1.0f - v);
        float r = fminf(1.f, 1.15f*__powf(v, 0.42f));
        float g = fminf(1.f, 0.95f*__powf(v, 0.56f));
        float b = fminf(1.f, 0.70f*__powf(v, 0.88f));
        return make_float3(r,g,b);
    }

    // 256 Threads/Block; mindestens 2 Blöcke/SM anpeilen
    __global__ __launch_bounds__(256,2)
    void perturbKernel(
        uchar4* __restrict__ out, int* __restrict__ iterOut,
        const float2* __restrict__ refOrbit, int maxIter,
        int w, int h, float zoom, float2 center,
        float2 c_ref, float tau_on2)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;
        const int idx = y*w + x;

        // 1) c (pro Pixel), dC = c - c_ref  — c_ref kommt als Kernel-Argument (einmalig auf Host berechnet)
        const float2 c  = pixelToComplex(x + 0.5, y + 0.5, w, h,
                                         (double)center.x, (double)center.y, (double)zoom);
        const float2 dC = make_float2(c.x - c_ref.x, c.y - c_ref.y);

        // Frühtest innen
        if (insideMainCardioidOrBulb(c.x, c.y)){
            out[idx]     = make_uchar4(0,0,0,255);
            iterOut[idx] = maxIter;
            return;
        }

        float2 delta = make_float2(0.f,0.f);
        int it = 0;
        const float esc2 = 4.0f;

        int exceedCount = 0; // Hysterese über 2 aufeinanderfolgende Schritte

        // 2) Perturbations-Loop
        for(int n=0; n<maxIter; ++n){
            // z_ref[n]: Broadcast-Pattern → L2/RO-Cache; __ldg optional (cc>=80 ohnehin cached)
            const float2 zref = refOrbit[n];

            // δ_{n+1} = 2*z_ref*δ + Δc
            const float2 twoZ = make_float2(2.f*zref.x, 2.f*zref.y);
            delta = f2_add(f2_mul(twoZ, delta), dC);

            // z ≈ z_ref + δ
            const float2 z  = f2_add(zref, delta);
            const float  z2 = f2_abs2(z);
            if (z2 > esc2){ it = n; goto ESCAPE; }

            // robuster, schneller Guard: |δ|^2 > tau_on^2 * max(|z|^2, eps)
            const float delta2 = f2_abs2(delta);
            if (delta2 > tau_on2 * fmaxf(z2, 1e-24f)){
                if (++exceedCount >= 2){
                    // 3) Fallback: direkte Iteration ab aktueller Schätzung (z)
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
            } else {
                exceedCount = 0;
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

    // 3b) c_ref EINMAL auf Host berechnen und als float2 übergeben
    //     (identisch zu der Berechnung im Kernel, aber konstant für alle Pixel)
    const float2 cref = pixelToComplex(0.5 + 0.5*w, 0.5 + 0.5*h, w, h,
                                       (double)offset.x, (double)offset.y, (double)zoom);

    // 4) Kernel
    const dim3 block(32, 8);  // 256 Threads → gute Occupancy, passend zu __launch_bounds__
    const dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    // Konservative, stabile Standardschwelle; tau_on^2 wird einmalig berechnet (kein sqrt im Kernel)
    const float deltaRelMax = 5e-3f;
    const float tau2 = deltaRelMax * deltaRelMax;

    perturbKernel<<<grid, block>>>(out, d_it, dRef, maxIter, w, h, zoom, offset, cref, tau2);
    if (!ok(cudaGetLastError(), "perturbKernel launch")) return;

    // 5) Perf-Log (optional)
    if (Settings::performanceLogging){
        if (!ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull kern=%.2f ms it=%d", ms, maxIter);
    }
}
