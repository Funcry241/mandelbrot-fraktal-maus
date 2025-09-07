///// Otter: Perturbation/Series Path – Referenz-Orbit (DD) + GPU-Delta; API unverändert.
///  + Zwei-Pass Shading: μ-Buffer -> Histogramm (256) -> CDF-LUT -> finaler Shade.
///  + Hash-Dither (kein Grid), Histogramm-Equalisierung reduziert Banding/Ringe massiv.
///// Schneefuchs: Deterministisch, ASCII-only; Mapping konsistent (center+zoom); kompakte PERF-Logs.
///  CUDA 13: __logf/__log2f/__powf/__fsqrt_rn/__fmaf_rn; Warp-lokales Histogramm.
///// Maus: Frühe Rückgaben; keine Overlays; keine CUDA_CHECK-Makros; Host-Wrapper bleibt launch_mandelbrotHybrid.
///  Datei: src/nacktmull.cu

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
#include "common.hpp"

// ------------------------------------------------------------
// Cardioid/Period-2-Bulb Test (Host/Device)
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
// Double-Double minimal (Host)
// ------------------------------------------------------------
namespace dd {
    struct dd { double hi, lo; };
    inline dd make(double x){ return {x, 0.0}; }
    inline dd two_sum(double a, double b){ double s=a+b; double bb=s-a; double e=(a-(s-bb))+(b-bb); return {s,e}; }
    inline dd quick_two_sum(double a,double b){ double s=a+b; double e=b-(s-a); return {s,e}; }
    inline dd add(dd a, dd b){ dd s=two_sum(a.hi,b.hi); double t=a.lo+b.lo+s.lo; return quick_two_sum(s.hi,t); }
    inline dd mul(dd a, dd b){
        double p=a.hi*b.hi; double e=std::fma(a.hi,b.hi,-p);
        e += a.hi*b.lo + a.lo*b.hi; return quick_two_sum(p,e);
    }
}

// ------------------------------------------------------------
// Referenz-Orbit (Host, DD) -> float2[maxIter]
// ------------------------------------------------------------
static void buildReferenceOrbitDD(std::vector<float2>& out, int maxIter, double cref_x, double cref_y){
    out.resize((size_t)maxIter);
    dd::dd zx=dd::make(0.0), zy=dd::make(0.0);
    dd::dd cr=dd::make(cref_x), ci=dd::make(cref_y);
    for(int i=0;i<maxIter;i++){
        dd::dd x2=dd::mul(zx,zx), y2=dd::mul(zy,zy), xy=dd::mul(zx,zy);
        dd::dd zr=dd::add(dd::add(x2, dd::make(-y2.hi)), cr);
        dd::dd zi=dd::add(dd::make(2.0*xy.hi), ci);
        zx=zr; zy=zi;
        out[(size_t)i]=make_float2((float)zx.hi,(float)zy.hi);
    }
}

// ------------------------------------------------------------
// Device helpers
// ------------------------------------------------------------
namespace {
    __device__ __forceinline__ float2 f2_add(float2 a,float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
    __device__ __forceinline__ float2 f2_mul(float2 a,float2 b){ return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
    __device__ __forceinline__ float  f2_abs2(float2 a){ return a.x*a.x + a.y*a.y; }

    // Hash-basiertes Pixel-Dither in [-amp,+amp] (deterministisch, kein Grid).
    __device__ __forceinline__ float pixel_dither(int x,int y,float amp=1.0f/256.0f){
        unsigned int h = (unsigned int)x * 0x9E3779B9u ^ (unsigned int)y * 0x85EBCA6Bu;
        h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
        float u = (float)(h & 0x00FFFFFFu) * (1.0f/16777216.0f);   // [0,1)
        return (u - 0.5f) * (2.0f*amp);
    }

    // Palette für μ in [0,1] (monoton, leicht kontrastbetont)
    __device__ __forceinline__ float3 shade_from_mu01(float mu01){
        float v = 1.f - __expf(-1.75f * mu01);
        v = __powf(fmaxf(0.f, v), 1.10f);           // mehr Steigung nahe 1 -> weniger Banding
        v = v + 0.04f * (1.0f - v);
        float r = fminf(1.f, 1.10f*__powf(v, 0.50f));
        float g = fminf(1.f, 0.96f*__powf(v, 0.70f));
        float b = fminf(1.f, 0.74f*__powf(v, 0.95f));
        return make_float3(r,g,b);
    }
}

// ============================================================
// Pass 1: μ- und it-Berechnung (Perturbation+Fallback), schreibt muBuf & iterOut
// ============================================================
__global__ void muKernel(
    float* __restrict__ muBuf, int* __restrict__ iterOut,
    const float2* __restrict__ refOrbit, int maxIter,
    int w, int h, float zoom, float2 center, float deltaRelMax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int idx = y*w + x;

    const float2 c     = pixelToComplex(x + 0.5, y + 0.5, w, h,
                                        (double)center.x, (double)center.y, (double)zoom);
    const float2 c_ref = pixelToComplex(0.5 + 0.5*w, 0.5 + 0.5*h, w, h,
                                        (double)center.x, (double)center.y, (double)zoom);
    const float2 dC = make_float2(c.x - c_ref.x, c.y - c_ref.y);

    if (insideMainCardioidOrBulb(c.x, c.y)){
        muBuf[idx]   = (float)maxIter;
        iterOut[idx] = maxIter;
        return;
    }

    float2 delta = make_float2(0.f,0.f);
    int   it = maxIter;
    float mu = (float)maxIter;
    const float esc2 = 4.0f;

    // Perturbations-Loop
    for(int n=0; n<maxIter; ++n){
        const float2 zref = refOrbit[n];
        const float2 twoZ = make_float2(2.f*zref.x, 2.f*zref.y);
        delta = f2_add(f2_mul(twoZ, delta), dC);

        const float2 z = f2_add(zref, delta);
        const float r2 = f2_abs2(z);
        if (r2 > esc2){
            const float log_r = 0.5f * __logf(r2);
            mu = (float)n + 1.0f - __log2f(fmaxf(1e-30f, log_r));
            it = n;
            goto DONE;
        }

        const float refMag = fmaxf(__fsqrt_rn(f2_abs2(zref)), 1e-12f);
        const float delMag = __fsqrt_rn(f2_abs2(delta));
        if (delMag > deltaRelMax * refMag){
            // Fallback: direkte Iteration
            float zx = z.x, zy = z.y;
            for(int m=n+1; m<maxIter; ++m){
                const float x2 = zx*zx, y2 = zy*zy;
                const float r2f = x2 + y2;
                if (r2f > 4.f){
                    const float log_r_f = 0.5f * __logf(r2f);
                    mu = (float)m + 1.0f - __log2f(fmaxf(1e-30f, log_r_f));
                    it = m;
                    goto DONE;
                }
                const float xt = x2 - y2 + c.x;
                zy = __fmaf_rn(2.f*zx, zy, c.y);
                zx = xt;
            }
            it = maxIter; mu = (float)maxIter; goto DONE;
        }
    }
    it = maxIter; mu = (float)maxIter;
DONE:
    muBuf[idx]   = mu;
    iterOut[idx] = it;
}

// ============================================================
// Pass 2: Histogramm über μ (256 Bins, warp-lokal -> global)
// ============================================================
namespace {
    constexpr int EN_BLOCK_THREADS = 128;
    constexpr int EN_BINS          = 256;
    constexpr int WARP_SIZE        = 32;
    constexpr int EN_WARPS         = EN_BLOCK_THREADS / WARP_SIZE;
}
__global__ __launch_bounds__(EN_BLOCK_THREADS)
void histKernel(const float* __restrict__ muBuf, int w, int h, int maxIter, unsigned int* __restrict__ gHist){
    __shared__ unsigned int sh[EN_WARPS][EN_BINS];
    const int lane = threadIdx.x & (WARP_SIZE-1);
    const int warp = threadIdx.x >> 5;

    // zero local
    for (int i=lane; i<EN_BINS; i+=WARP_SIZE) sh[warp][i]=0u;
    __syncthreads();

    const int N = w*h;
    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < N; i += blockDim.x*gridDim.x){
        float mu = muBuf[i];
        float mu01 = fminf(1.0f, fmaxf(0.0f, mu / (float)maxIter));
        int bin = (int)(mu01 * 255.0f);
        bin = (bin < 0) ? 0 : (bin > 255 ? 255 : bin);
        atomicAdd(&sh[warp][bin], 1u);
    }
    __syncthreads();

    if (threadIdx.x < EN_BINS){
        unsigned int sum = 0u;
        #pragma unroll
        for (int widx=0; widx<EN_WARPS; ++widx) sum += sh[widx][threadIdx.x];
        atomicAdd(&gHist[threadIdx.x], sum);
    }
}

// Constant memory LUT (256 floats)
__device__ __constant__ float gEqualizeLUT[256];

// ============================================================
// Pass 3: Shading – LUT-Equalisierung + Dither -> RGBA
// ============================================================
__global__ void shadeKernel(
    const float* __restrict__ muBuf, uchar4* __restrict__ out,
    int w, int h, int maxIter)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int idx = y*w + x;

    float mu = muBuf[idx];
    float mu01 = fminf(1.0f, fmaxf(0.0f, mu / (float)maxIter));

    // LUT Equalisierung (binäre Zuordnung, optional: linear Interpolation)
    int bin = (int)(mu01 * 255.0f);
    bin = (bin < 0) ? 0 : (bin > 255 ? 255 : bin);
    float eq = gEqualizeLUT[bin];

    // Dither (kein Grid), kleine Amplitude
    eq = fminf(1.0f, fmaxf(0.0f, eq + pixel_dither(x, y, 1.0f/256.0f)));

    const float3 col = shade_from_mu01(eq);
    out[idx] = make_uchar4((unsigned char)(255.f*col.x),
                           (unsigned char)(255.f*col.y),
                           (unsigned char)(255.f*col.z), 255);
}

// ============================================================
// Öffentliche API – orchestriert die 3 Passes
// ============================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    auto ok = [](cudaError_t e, const char* op)->bool {
        if (e != cudaSuccess) { LUCHS_LOG_HOST("[CUDA][ERR] %s failed: %d (%s)", op, int(e), cudaGetErrorString(e)); return false; }
        return true;
    };

    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();

    if (!out || !d_it || w <= 0 || h <= 0 || maxIter <= 0) {
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",
                       (void*)out, (void*)d_it, w, h, maxIter);
        return;
    }

    // 1) Referenz-Orbit (Host, DD)
    const double cref_x = (double)offset.x;
    const double cref_y = (double)offset.y;
    static std::vector<float2> hRef;
    try { buildReferenceOrbitDD(hRef, maxIter, cref_x, cref_y); }
    catch(...) { LUCHS_LOG_HOST("[NACKTMULL][ERR] buildReferenceOrbitDD threw; aborting frame"); return; }

    // Device Puffer (only-grow)
    static float2* dRef = nullptr; static size_t dRefCap = 0;
    static float*  dMu  = nullptr; static size_t dMuCap  = 0;
    static unsigned int* dHist = nullptr; static size_t dHistCap = 0;

    const size_t orbitBytes = (size_t)maxIter * sizeof(float2);
    if (orbitBytes > dRefCap){
        if (dRef && !ok(cudaFree(dRef), "cudaFree(dRef)")) { dRef=nullptr; dRefCap=0; return; }
        if (!ok(cudaMalloc(&dRef, orbitBytes), "cudaMalloc(dRef)")) return;
        dRefCap = orbitBytes;
    }
    if (!ok(cudaMemcpy(dRef, hRef.data(), orbitBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D(ref)")) return;

    const size_t pixels = (size_t)w * (size_t)h;
    const size_t muBytes = pixels * sizeof(float);
    if (muBytes > dMuCap){
        if (dMu && !ok(cudaFree(dMu), "cudaFree(dMu)")) { dMu=nullptr; dMuCap=0; return; }
        if (!ok(cudaMalloc(&dMu, muBytes), "cudaMalloc(dMu)")) return;
        dMuCap = muBytes;
    }

    if (256 * sizeof(unsigned int) > dHistCap){
        if (dHist && !ok(cudaFree(dHist), "cudaFree(dHist)")) { dHist=nullptr; dHistCap=0; return; }
        if (!ok(cudaMalloc(&dHist, 256 * sizeof(unsigned int)), "cudaMalloc(dHist)")) return;
        dHistCap = 256 * sizeof(unsigned int);
    }

    // 2) Pass 1 – μ+it
    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);
    const float deltaRelMax = 1e-3f;

    muKernel<<<grid, block>>>(dMu, d_it, dRef, maxIter, w, h, zoom, offset, deltaRelMax);
    if (!ok(cudaGetLastError(), "muKernel")) return;

    // 3) Pass 2 – Histogramm (global 256 Bins = 1024 Bytes)
    if (!ok(cudaMemset(dHist, 0, 256 * sizeof(unsigned int)), "cudaMemset(hist)")) return;

    // genug Blöcke um Vollauslastung zu erreichen
    const int histBlocks = (int)std::min<size_t>(  (pixels + EN_BLOCK_THREADS - 1) / EN_BLOCK_THREADS,
                                                   8u * 1024u ); // Obergrenze
    histKernel<<<histBlocks, EN_BLOCK_THREADS>>>(dMu, w, h, maxIter, dHist);
    if (!ok(cudaGetLastError(), "histKernel")) return;

    // Histogramm -> Host, CDF -> LUT
    unsigned int hHist[256];
    if (!ok(cudaMemcpy(hHist, dHist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H(hist)")) return;

    // CDF
    unsigned long long total = 0ULL;
    for (int i=0;i<256;++i) total += (unsigned long long)hHist[i];
    float lut[256];
    if (total == 0ULL) {
        for (int i=0;i<256;++i) lut[i] = (float)i/255.0f;
    } else {
        unsigned long long run = 0ULL;
        for (int i=0;i<256;++i){
            run += (unsigned long long)hHist[i];
            float cdf = (float)run / (float)total;
            // leichtes Clipping gegen extreme Flächen, glättet harte Bänder
            const float epsLo = 0.005f, epsHi = 0.995f;
            lut[i] = (cdf - epsLo) / (epsHi - epsLo);
            if (lut[i] < 0.0f) lut[i] = 0.0f;
            if (lut[i] > 1.0f) lut[i] = 1.0f;
        }
    }
    if (!ok(cudaMemcpyToSymbol(gEqualizeLUT, lut, 256*sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol(LUT)")) return;

    // 4) Pass 3 – Shading
    shadeKernel<<<grid, block>>>(dMu, out, w, h, maxIter);
    if (!ok(cudaGetLastError(), "shadeKernel")) return;

    if (Settings::performanceLogging){
        if (!ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return;
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        LUCHS_LOG_HOST("[PERF] nacktmull μ+hist+shade=%.2f ms it=%d", ms, maxIter);
    }
}
