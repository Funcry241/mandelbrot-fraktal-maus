///// Otter: Nacktmull + HEP-PoC – Histogram-Equalized Palette (256 Bins) ohne neue Dateien; continuous iteration (sqrt-free), deterministisches Mini-Dither.
/// ///// Schneefuchs: Deterministisch, ASCII-only; Mapping (center+zoom) konsistent; keine CUDA_CHECKs hier; kompakte [PERF]-Logs.
/// ///// Maus: Minimal-invasiv: perturbKernel schreibt it; danach HEP-Histogramm + CDF + Recolor in dieser Datei.
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
// GPU-Kerne & Helfer (Perturbation + HEP-PoC)
// δ_{n+1} = 2*z_ref[n]*δ_n + Δc;   z ≈ z_ref[n] + δ
// Escape, wenn |z|^2 > 4. Danach HEP-Histogramm + CDF + Recolor.
// ------------------------------------------------------------
namespace {
    __device__ __forceinline__ float2 f2_add(float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
    __device__ __forceinline__ float2 f2_mul(float2 a, float2 b){ return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
    __device__ __forceinline__ float  f2_abs2(float2 a){ return a.x*a.x + a.y*a.y; }

    // Glättende Kurve von t -> RGB
    __device__ __forceinline__ float3 shade_from_t(float t){
        t = fminf(1.0f, fmaxf(0.0f, t));
        float v = 1.f - __expf(-1.75f * t);
        v = powf(fmaxf(0.f, v), 0.80f);
        v = v + 0.08f * (1.0f - v);
        float r = fminf(1.f, 1.15f*powf(v, 0.42f));
        float g = fminf(1.f, 0.95f*powf(v, 0.56f));
        float b = fminf(1.f, 0.70f*powf(v, 0.88f));
        return make_float3(r,g,b);
    }

    // Kleines, deterministisches Dither in [0,1)
    __device__ __forceinline__ unsigned int wanghash(unsigned int x){
        x = (x ^ 61u) ^ (x >> 16);
        x *= 9u;
        x = x ^ (x >> 4);
        x *= 0x27d4eb2du;
        x = x ^ (x >> 15);
        return x;
    }
    __device__ __forceinline__ float jitter01(unsigned int seed){
        return (float)((wanghash(seed) >> 8) * (1.0f/16777216.0f)); // 24-bit / 2^24
    }

    // ------------------------ Mandelbrot Perturbation ------------------------
    __global__ void perturbKernel(
        uchar4* __restrict__ out, int* __restrict__ iterOut,
        const float2* __restrict__ refOrbit, int maxIter,
        int w, int h, float zoom, float2 center, float deltaRelMax)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;
        const int idx = y*w + x;

        // Konsistentes Mapping: center+zoom (wie in restlicher Pipeline)
        const float2 c     = pixelToComplex(x + 0.5, y + 0.5, w, h,
                                            (double)center.x, (double)center.y, (double)zoom);
        const float2 c_ref = pixelToComplex(0.5 + 0.5*w, 0.5 + 0.5*h, w, h,
                                            (double)center.x, (double)center.y, (double)zoom);
        const float2 dC = make_float2(c.x - c_ref.x, c.y - c_ref.y);

        if (insideMainCardioidOrBulb(c.x, c.y)){
            out[idx] = make_uchar4(0,0,0,255);
            iterOut[idx] = maxIter;
            return;
        }

        float2 delta = make_float2(0.f,0.f);
        int it = 0;
        const float esc2 = 4.0f;
        const float deltaRelMax2 = deltaRelMax * deltaRelMax; // Vergleich auf Quadrate (schneller)

        for(int n=0; n<maxIter; ++n){
            const float2 zref = refOrbit[n];
            const float2 twoZ = make_float2(2.f*zref.x, 2.f*zref.y);
            delta = f2_add(f2_mul(twoZ, delta), dC);  // δ_{n+1}

            const float2 z = f2_add(zref, delta);
            const float r2 = f2_abs2(z);
            if (r2 > esc2){ it = n;
                // Continuous Iteration (sqrt-frei): nu = n + 2 - log2( ln(|z|^2) )
                float nu = (float)n + 2.0f - __log2f(__logf(r2));
                float t  = fminf(1.f, fmaxf(0.f, nu / (float)maxIter));
                t += (jitter01((unsigned int)idx ^ (unsigned int)(n*747796405u)) - 0.5f) * (1.0f / (float)maxIter);
                const float3 col = shade_from_t(t);
                out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                       (unsigned char)(255.f*col.y),
                                       (unsigned char)(255.f*col.z), 255);
                iterOut[idx] = it; // Roh-Iterationszahl für HEP
                return;
            }

            // Fallback-Schwelle ohne sqrtf: |δ|^2 > (δ_max^2) * max(|z_ref|^2, ε^2)
            const float ref2 = fmaxf(f2_abs2(zref), 1e-24f);
            const float del2 = f2_abs2(delta);
            if (del2 > deltaRelMax2 * ref2){
                // Fallback: direkte Iteration ab aktuellem Zustand
                float zx = z.x, zy = z.y;
                for(int m=n+1; m<maxIter; ++m){
                    const float x2 = zx*zx, y2 = zy*zy;
                    if (x2 + y2 > 4.f){ it = m;
                        float r2m = x2 + y2;
                        float nu  = (float)m + 2.0f - __log2f(__logf(r2m));
                        float t   = fminf(1.f, fmaxf(0.f, nu / (float)maxIter));
                        t += (jitter01((unsigned int)idx ^ (unsigned int)(m*747796405u)) - 0.5f) * (1.0f / (float)maxIter);
                        const float3 col = shade_from_t(t);
                        out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                               (unsigned char)(255.f*col.y),
                                               (unsigned char)(255.f*col.z), 255);
                        iterOut[idx] = it;
                        return;
                    }
                    const float xt = x2 - y2 + c.x; zy = 2.f*zx*zy + c.y; zx = xt;
                }
                it = maxIter;
                const float3 col = shade_from_t(1.f);
                out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                       (unsigned char)(255.f*col.y),
                                       (unsigned char)(255.f*col.z), 255);
                iterOut[idx] = it;
                return;
            }
        }
        // Nicht entkommen
        it = maxIter;
        {
            const float3 col = shade_from_t(1.f);
            out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                   (unsigned char)(255.f*col.y),
                                   (unsigned char)(255.f*col.z), 255);
            iterOut[idx] = it;
        }
    }

    // ------------------------ HEP-PoC: Histogram, CDF, Recolor ------------------------
    namespace hep {
        constexpr int BINS = 256;
        constexpr int BLOCK_THREADS = 256;

        __device__ __forceinline__ int it_to_bin(int it, int maxIter){
            if (it < 0) it = 0;
            // skaliere it in [0,255]
            int bin = (int)(( (unsigned long long)it * BINS ) / (unsigned long long)(maxIter + 1));
            return bin > (BINS-1) ? (BINS-1) : bin;
        }

        // Einfache, robuste Histogramm-Kernel (block-lokal -> global)
        __global__ void histKernel(const int* __restrict__ it, unsigned int* __restrict__ gHist,
                                   int w, int h, int maxIter)
        {
            __shared__ unsigned int sHist[BINS];
            for (int i = threadIdx.x; i < BINS; i += blockDim.x) sHist[i] = 0u;
            __syncthreads();

            const size_t total = (size_t)w * (size_t)h;
            for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
                 idx < total; idx += (size_t)gridDim.x * blockDim.x)
            {
                int v = it[idx];
                int b = it_to_bin(v, maxIter);
                atomicAdd(&sHist[b], 1u);
            }
            __syncthreads();

            for (int i = threadIdx.x; i < BINS; i += blockDim.x)
                atomicAdd(&gHist[i], sHist[i]);
        }

        // CDF + Normalisierung -> LUT[256] in [0,1]
        __global__ void cdfKernel(const unsigned int* __restrict__ gHist,
                                  float* __restrict__ lut, unsigned int total)
        {
            __shared__ unsigned int scan[BINS];
            const int tid = threadIdx.x;
            if (tid < BINS) scan[tid] = gHist[tid];
            __syncthreads();

            if (tid == 0) {
                unsigned long long acc = 0ull;
                #pragma unroll
                for (int i=0;i<BINS;i++){ acc += (unsigned long long)scan[i]; scan[i] = (unsigned int)acc; }
            }
            __syncthreads();

            if (tid < BINS) {
                float cdf = (total > 0u) ? ((float)scan[tid] / (float)total) : 0.0f;
                lut[tid] = fminf(1.0f, cdf);
            }
        }

        // Recolor: liest Iterationen + LUT und überschreibt out[]
        __global__ void recolorKernel(uchar4* __restrict__ out,
                                      const int* __restrict__ it,
                                      const float* __restrict__ lut,
                                      int w, int h, int maxIter)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= w || y >= h) return;
            const int idx = y*w + x;

            int v = it[idx];
            const int bin = it_to_bin(v, maxIter);
            float t = lut[bin];

            // Mini-Dither, sehr zart, um Restbanding zu brechen (deterministisch)
            t += (jitter01((unsigned int)idx * 747796405u) - 0.5f) * (1.0f / (float)(maxIter > 0 ? maxIter : 1));
            const float3 col = shade_from_t(t);

            out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                                   (unsigned char)(255.f*col.y),
                                   (unsigned char)(255.f*col.z), 255);
        }
    } // namespace hep
} // anonym

// ============================================================
// Öffentliche API – ersetzt alte Hybrid-Funktion
// Signatur bleibt identisch: launch_mandelbrotHybrid(...)
// Keine Exceptions (extern "C"); eigene Fehlerprüfung + early return.
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

    // 4) Kernel: Mandelbrot (Perturbation)
    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);
    const float deltaRelMax = 1e-3f; // Startwert; ggf. später adaptiv

    (void)cudaGetLastError(); // clear sticky
    perturbKernel<<<grid, block>>>(out, d_it, dRef, maxIter, w, h, zoom, offset, deltaRelMax);
    cudaError_t kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA][ERR] perturbKernel launch: %d (%s)", int(kernErr), cudaGetErrorString(kernErr));
        return;
    }

    // 5) HEP-PoC: Histogram (256), CDF -> LUT, Recolor (überschreibt out[])
    const size_t totalPx = (size_t)w * (size_t)h;
    static unsigned int* dHist = nullptr;
    static float*        dLut  = nullptr;
    static size_t        capH  = 0, capL = 0;

    const size_t needHist = 256 * sizeof(unsigned int);
    const size_t needLut  = 256 * sizeof(float);
    if (needHist > capH) { if (dHist) { if (!ok(cudaFree(dHist), "cudaFree(dHist)")) { dHist=nullptr; capH=0; return; } } }
    if (!dHist) { if (!ok(cudaMalloc(&dHist, needHist), "cudaMalloc(dHist)")) return; capH = needHist; }
    if (needLut > capL)  { if (dLut)  { if (!ok(cudaFree(dLut),  "cudaFree(dLut)"))  { dLut=nullptr;  capL=0;  return; } } }
    if (!dLut)  { if (!ok(cudaMalloc(&dLut, needLut),  "cudaMalloc(dLut)"))  return; capL = needLut; }

    if (!ok(cudaMemset(dHist, 0, needHist), "cudaMemset(dHist)")) return;

    // Histogramm: moderate Blockzahl mit Stride-Schleife im Kernel
    const int heThreads = hep::BLOCK_THREADS;
    int heBlocks = (int)((totalPx + (size_t)heThreads - 1) / (size_t)heThreads);
    if (heBlocks > 32768) heBlocks = 32768; // Deckel
    hep::histKernel<<<heBlocks, heThreads>>>(d_it, dHist, w, h, maxIter);
    kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA][ERR] hep::histKernel: %d (%s)", int(kernErr), cudaGetErrorString(kernErr));
        return;
    }

    hep::cdfKernel<<<1, hep::BINS>>>(dHist, dLut, (unsigned int)totalPx);
    kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA][ERR] hep::cdfKernel: %d (%s)", int(kernErr), cudaGetErrorString(kernErr));
        return;
    }

    hep::recolorKernel<<<grid, block>>>(out, d_it, dLut, w, h, maxIter);
    kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA][ERR] hep::recolorKernel: %d (%s)", int(kernErr), cudaGetErrorString(kernErr));
        return;
    }

    // 6) Perf-Log (optional)
    if (Settings::performanceLogging){
        if (!ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull kern+hep=%.2f ms it=%d", ms, maxIter);
    }
}
