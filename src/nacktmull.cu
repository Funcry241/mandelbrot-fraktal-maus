// =============================================================================
// Projekt Nacktmull – Perturbation/Series Path (ersetzt Hybrid-Pipeline)
// Datei: src/nacktmull.cu
//
// Ziel:
//   * Tiefes Zoomen via Referenz-Orbit (High-Precision am Host) +
//     GPU-Perturbation (linearisierte Δ-Rekurrenz).
//   * Keine Overlays/Sprites; reine Mathematik.
//   * Schlanke, ASCII-only PERF-Logs (Zeit in ms; Host logger liefert EPOCH-MILLIS).
//
// Hinweise:
//   * Erste lauffähige Stufe: Referenz in Double‑Double (software) → nach float2
//     komprimiert und an GPU übergeben. Später austauschbar gegen echtes MP.
//   * Kernel besitzt Fallback auf direkte Iteration, wenn Δ zu groß wird.
//   * API bleibt kompatibel: exportierte Funktion heißt launch_mandelbrotHybrid(...)
//     und ersetzt die alte Implementierung – Call‑Sites bleiben unverändert.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "nacktmull_math.cuh"
#include "common.hpp"

// ------------------------------------------------------------
// Lokale Helpers (unabhängig von core_kernel)
// ------------------------------------------------------------
namespace {
    struct float2x { float x, y; };

    __host__ __device__ inline float2x make_f2(float x, float y){ return {x,y}; }

    __host__ __device__ inline float2x cadd(const float2x&a,const float2x&b){return {a.x+b.x,a.y+b.y};}
    __host__ __device__ inline float2x cmul(const float2x&a,const float2x&b){return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};}
    __host__ __device__ inline float  cabs2(const float2x&a){return a.x*a.x + a.y*a.y;}

    // Mapping Pixel → C (wie zuvor)
    __host__ __device__ inline float2x pixelToC(int ix,int iy,int w,int h,float zoom, float2x offset){
        float scale = 1.0f / zoom;
        float spanX = 3.5f * scale;
        float spanY = spanX * (float)h / (float)w;
        float cx = ((ix + 0.5f) / (float)w - 0.5f) * spanX + offset.x;
        float cy = ((iy + 0.5f) / (float)h - 0.5f) * spanY + offset.y;
        return {cx, cy};
    }

    __host__ __device__ inline bool insideMainCardioidOrBulb(float x, float y){
        float xm = x - 0.25f;
        float q  = xm * xm + y * y;
        if (q * (q + xm) <= 0.25f * y * y) return true; // main cardioid
        float xp = x + 1.0f;
        if (xp * xp + y * y <= 0.0625f) return true;    // period-2 bulb
        return false;
    }
}

// ------------------------------------------------------------
// Double‑Double – minimal (nur add/mul) für Referenz‑Orbit am Host
// ------------------------------------------------------------
namespace dd {
    struct dd { double hi, lo; }; // hi+lo mit |lo| << |hi|

    inline dd make(double x){ return {x, 0.0}; }

    // Dekker/Veltkamp‑basics
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
        // FMA verbessert Genauigkeit, wenn vorhanden
        double p = a.hi * b.hi;
        double e = std::fma(a.hi, b.hi, -p);
        e += a.hi * b.lo + a.lo * b.hi;
        return quick_two_sum(p, e);
    }
}

// ------------------------------------------------------------
// Referenz‑Orbit: z_{n+1} = z_n^2 + c_ref  (Host, DD)
// Ergebnis als float2 gespeichert (kompakt), Länge = maxIter
// ------------------------------------------------------------
static void buildReferenceOrbitDD(std::vector<float2>& out, int maxIter, double cref_x, double cref_y){
    out.resize((size_t)maxIter);
    dd::dd zx = dd::make(0.0), zy = dd::make(0.0);
    dd::dd cr = dd::make(cref_x), ci = dd::make(cref_y);

    for(int i=0;i<maxIter;i++){
        // z^2
        dd::dd x2 = dd::mul(zx, zx);
        dd::dd y2 = dd::mul(zy, zy);
        dd::dd xy = dd::mul(zx, zy);
        dd::dd zr = dd::add(dd::add(x2, dd::make(-y2.hi)), cr); // (x^2 - y^2) + cr  (lo grob vernachlässigt)
        dd::dd zi = dd::add(dd::add(dd::make(2.0*xy.hi), dd::make(0.0)), ci);
        zx = zr; zy = zi;
        out[(size_t)i] = make_float2((float)zx.hi, (float)zy.hi);
    }
}

// ------------------------------------------------------------
// GPU‑Kernel: Perturbation
// δ_{n+1} = 2*z_ref[n]*δ_n + Δc;   z ≈ z_ref[n] + δ
// Escape, wenn |z|^2 > 4.
// Fallback: direkte Iteration ab aktuellem Zustand, falls |δ| zu groß.
// ------------------------------------------------------------
namespace {
    __device__ __forceinline__ float2 f2_add(float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
    __device__ __forceinline__ float2 f2_mul(float2 a, float2 b){ return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
    __device__ __forceinline__ float  f2_abs2(float2 a){ return a.x*a.x + a.y*a.y; }

    __device__ inline float3 shade_from_iter(int it, int maxIter){
        // simple smooth coloring
        float t = fminf(1.0f, (float)it / (float)maxIter);
        float v = 1.f - __expf(-1.75f * t);
        v = powf(fmaxf(0.f, v), 0.80f);
        v = v + 0.08f * (1.0f - v);
        float r = fminf(1.f, 1.15f*powf(v, 0.42f));
        float g = fminf(1.f, 0.95f*powf(v, 0.56f));
        float b = fminf(1.f, 0.70f*powf(v, 0.88f));
        return make_float3(r,g,b);
    }

    __global__ void perturbKernel(
        uchar4* __restrict__ out, int* __restrict__ iterOut,
        const float2* __restrict__ refOrbit, int maxIter,
        int w, int h, float zoom, float2 offset, float deltaRelMax)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;
        int idx = y*w + x;

        // Rechenzentrum = Referenzpunkt: Bildschirmmitte
        float2 c = pixelToComplex(x + 0.5f, y + 0.5f, w, h,
                                  3.5f*(1.f/zoom), 3.5f*(1.f/zoom)*(float)h/(float)w, offset);
        float2 c_ref = pixelToComplex((w/2)+0.5f, (h/2)+0.5f, w, h,
                                      3.5f*(1.f/zoom), 3.5f*(1.f/zoom)*(float)h/(float)w, offset);
        float2 dC = make_float2(c.x - c_ref.x, c.y - c_ref.y);

        if (insideMainCardioidOrBulb(c.x, c.y)){
            out[idx] = make_uchar4(0,0,0,255);
            iterOut[idx] = maxIter;
            return;
        }

        // linearisierte Δ‑Rekurrenz
        float2 delta = make_float2(0.f,0.f);
        int it = 0;
        const float esc2 = 4.0f;

        for(int n=0; n<maxIter; ++n){
            float2 zref = refOrbit[n];
            // δ_{n+1} = 2*zref*δ + Δc
            float2 twoZ = make_float2(2.f*zref.x, 2.f*zref.y);
            delta = f2_add(f2_mul(twoZ, delta), dC);

            float2 z = f2_add(zref, delta);
            float r2 = f2_abs2(z);
            if (r2 > esc2){ it = n; goto ESCAPE; }

            // Stabilität prüfen
            float refMag = fmaxf(sqrtf(f2_abs2(zref)), 1e-12f);
            float delMag = sqrtf(f2_abs2(delta));
            if (delMag > deltaRelMax * refMag){
                // Fallback: direkte Iteration ab aktuellem Zustand
                float zx = z.x, zy = z.y;
                for(int m=n+1; m<maxIter; ++m){
                    float x2 = zx*zx, y2 = zy*zy;
                    if (x2 + y2 > 4.f){ it = m; goto ESCAPE; }
                    float xt = x2 - y2 + c.x; zy = 2.f*zx*zy + c.y; zx = xt;
                }
                it = maxIter; goto ESCAPE;
            }
        }
        it = maxIter;
    ESCAPE:
        {
            // Farbe aus Iteration
            float3 col = shade_from_iter(it, maxIter);
            out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                   (unsigned char)(255.f*col.y),
                                   (unsigned char)(255.f*col.z), 255);
            iterOut[idx] = it;
        }
    }
}

// ============================================================
// Öffentliche API – ersetzt alte Hybrid‑Funktion
// Signatur bleibt identisch zu vorherigem launch_mandelbrotHybrid(...)
// ============================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    // 1) Referenzpunkt = Bildschirmmitte
    const float scale = 1.0f / zoom;
    const float spanX = 3.5f * scale;
    const float spanY = spanX * (float)h / (float)w;
    const double cref_x = ((w*0.5 + 0.5) / (double)w - 0.5) * (double)spanX + (double)offset.x;
    const double cref_y = ((h*0.5 + 0.5) / (double)h - 0.5) * (double)spanY + (double)offset.y;

    // 2) Referenz‑Orbit (Host, DD)
    static std::vector<float2> hRef;
    buildReferenceOrbitDD(hRef, maxIter, cref_x, cref_y);

    // 3) Device‑Puffer verwalten (only‑grow)
    static float2* dRef = nullptr; static size_t dCap = 0;
    const size_t need = (size_t)maxIter * sizeof(float2);
    if (need > dCap){ if (dRef) cudaFree(dRef); cudaMalloc(&dRef, need); dCap = need; }
    cudaMemcpy(dRef, hRef.data(), need, cudaMemcpyHostToDevice);

    // 4) Kernel starten
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);
    const float deltaRelMax = 1e-3f; // Start‑Schranke, später adaptiv

    perturbKernel<<<grid, block>>>(out, d_it, dRef, maxIter, w, h, zoom, offset, deltaRelMax);

    // 5) (optionales) Budget/Perf‑Log – kompakt
    if (Settings::performanceLogging){
        cudaDeviceSynchronize();
        double ms = std::chrono::duration<double, std::milli>(clk::now()-t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull kern=%.2f ms it=%d", ms, maxIter);
    }
}
