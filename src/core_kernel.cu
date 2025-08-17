// core_kernel.cu — 2-Pass Mandelbrot (Warmup + Survivor Finish)
// 🐭 Maus: Kern schlank; ASCII-Logs bleiben deterministisch.
// 🦦 Otter: Smooth Coloring + Paletten (otter_color.hpp).
// 🦊 Schneefuchs: Warp-synchron, CHUNKed, reduzierte Divergenz + kompakte Survivor-Liste.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include <chrono>
#include "common.hpp"
#include "luchs_device_format.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"
#include "otter_color.hpp"

// ---------- Tuning -----------------------------------------------------------
namespace {
    // Größerer CHUNK → weniger Ballots, gleiche Bildqualität.
    constexpr int   WARP_CHUNK        = 32;

    // Seltene Periodizitätsprobe (für Pass 2)
    constexpr int   LOOP_CHECK_EVERY  = 32;
    constexpr float LOOP_EPS2         = 5e-8f;  // konservativ (close_hits>=2 gefordert)

    // 2-Pass: Warmup-Grenze (Escaper werden sofort gefärbt, Survivors kompakt gesammelt)
    constexpr int   WARMUP_IT         = 1024;

    // Otter: Paletten-/Shading-Defaults
    constexpr otter::Palette kPalette   = otter::Palette::Glacier; // Aurora/Glacier/Ember
    constexpr float          kStripeF   = 3.0f;
    constexpr float          kStripeAmp = 0.10f;
    constexpr float          kGamma     = 2.2f;
}

// ---------- Helpers ----------------------------------------------------------
__device__ __forceinline__ float2 pixelToComplex(
    float px, float py, int w, int h,
    float spanX, float spanY, float2 offset)
{
    return make_float2(
        (px / w - 0.5f) * spanX + offset.x,
        (py / h - 0.5f) * spanY + offset.y
    );
}

__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y) {
    // Hauptcardioide
    float xm = x - 0.25f;
    float q  = xm * xm + y * y;
    if (q * (q + xm) <= 0.25f * y * y) return true;
    // period-2 Bulb um (-1,0), r=0.25
    float xp = x + 1.0f;
    if (xp * xp + y * y <= 0.0625f) return true;
    return false;
}

// ---------- Iteration (CHUNKed) ---------------------------------------------
// Pass 1: Warmup ohne Periodizitätsprobe (nur Escape-Check)
__device__ __forceinline__ int iterate_warmup_noLoop(
    float cr, float ci, int maxSteps, float& x, float& y)
{
    float xx=0.0f, yy=0.0f, xy=0.0f;
    x=0.0f; y=0.0f;
    int it = 0;

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    bool active = true;

#pragma unroll 1
    for (int k=0; k<maxSteps; k+=WARP_CHUNK) {
#pragma unroll 1
        for (int s=0; s<WARP_CHUNK; ++s) {
            if (!active) continue;

            xx = x * x;
            yy = y * y;
            if (xx + yy > 4.0f) { active = false; continue; }

            xy = x * y;
            float xt = fmaf(x, x, -yy) + cr;   // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);         // 2*x*y + ci
            x = xt;
            ++it;

            if (it >= maxSteps) break;
        }
        unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }
    return it;
}

// Pass 2: Finish mit seltener Periodizitätsprobe (robust, konservativ)
__device__ __forceinline__ int iterate_finish_loopcheck(
    float cr, float ci, int start_it, int maxIter, float& x, float& y)
{
    int it = start_it;

    // Für seltene Periodizitätsprobe
    float px = x, py = y;
    int   pc = 0;
    int   close_hits = 0;

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    bool active = true;

    const int remain = maxIter - start_it;

#pragma unroll 1
    for (int k=0; k<remain; k+=WARP_CHUNK) {
#pragma unroll 1
        for (int s=0; s<WARP_CHUNK; ++s) {
            if (!active) { ++pc; continue; }

            float x2 = x * x;
            float y2 = y * y;
            if (x2 + y2 > 4.0f) { active = false; ++pc; continue; }

            float xt = fmaf(x, x, -y2) + cr;   // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);
            x = xt;
            ++it;
            ++pc;

            if (pc >= LOOP_CHECK_EVERY) {
                float dx = x - px, dy = y - py;
                float d2 = dx * dx + dy * dy;
                if (d2 < LOOP_EPS2) {
                    if (++close_hits >= 2) {
                        // sehr stabil → „innen“
                        active = false;
                        it = maxIter;
                    }
                } else {
                    close_hits = 0;
                }
                px = x; py = y; pc = 0;
            }
            if (it >= maxIter) break;
        }
        unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }
    return it;
}

// ---------- Survivor-Payload -------------------------------------------------
struct Survivor {
    float x, y;    // aktuelles z
    float cr, ci;  // konstantes c
    int   it;      // bisherige Iterationen (WARMUP_IT)
    int   idx;     // Pixelindex
};

// ---------- Kernel: Pass 1 (Warmup + Kompaktierung) -------------------------
__global__ __launch_bounds__(256, 2)
void mandelbrotPass1Warmup(
    uchar4* out, int* iterOut,
    Survivor* surv, int* survCount,
    int w, int h, float zoom, float2 offset,
    int maxIter)
{
    const bool doLog = Settings::debugLogging;

    const int xPix = blockIdx.x * blockDim.x + threadIdx.x;
    const int yPix = blockIdx.y * blockDim.y + threadIdx.y;
    if (xPix >= w || yPix >= h) return;

    const int idx = yPix * w + xPix;

    const float scale = 1.0f / zoom;
    const float spanX = 3.5f * scale;
    const float spanY = spanX * (float)h / (float)w;
    const float2 c = pixelToComplex(xPix + 0.5f, yPix + 0.5f, w, h, spanX, spanY, offset);

    // 🐽 Schwarze Schnauze (Innenpunkte sofort)
    if (insideMainCardioidOrBulb(c.x, c.y)) {
        out[idx]   = make_uchar4(0,0,0,255);
        iterOut[idx] = maxIter;
        if (doLog && threadIdx.x==0 && threadIdx.y==0) {
            char msg[96]; int n=0;
            n = luchs::d_append_str(msg,sizeof(msg),n,"[NOSE] early_inside x=");
            n = luchs::d_append_int(msg,sizeof(msg),n,xPix);
            n = luchs::d_append_str(msg,sizeof(msg),n," y=");
            n = luchs::d_append_int(msg,sizeof(msg),n,yPix);
            luchs::d_terminate(msg,sizeof(msg),n);
            LUCHS_LOG_DEVICE(msg);
        }
        return;
    }

    // Warmup bis WARMUP_IT
    float zx=0.0f, zy=0.0f;
    int itWarm = iterate_warmup_noLoop(c.x, c.y, WARMUP_IT, zx, zy);

    const float norm = zx*zx + zy*zy;
    const bool escaped = (itWarm < WARMUP_IT) && (norm > 4.0f);

    if (escaped) {
        // Otter-Färbung
        float3 col = otter::shade(itWarm, maxIter, zx, zy, kPalette, kStripeF, kStripeAmp, kGamma);
        out[idx] = make_uchar4(
            (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
            255);
        iterOut[idx] = itWarm;
        return;
    }

    // Survivor → warp-aggregated kompakter Push
    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    const bool isSurvivor = true;
    const unsigned ballot = __ballot_sync(mask, isSurvivor);
    const int voteCount   = __popc(ballot);
    const int lane        = int(threadIdx.x) & 31;
    const unsigned laneMask = ballot & ((1u << lane) - 1u);
    const int prefix      = __popc(laneMask);

    int base = 0;
    if (lane == 0) {
        base = atomicAdd(survCount, voteCount);
    }
    base = __shfl_sync(mask, base, 0);

    // schreiben
    Survivor s;
    s.x = zx; s.y = zy; s.cr = c.x; s.ci = c.y; s.it = itWarm; s.idx = idx;
    surv[base + prefix] = s;

    // (kein Ausgabeschreiben hier—Pass 2 kümmert sich)
}

// ---------- Kernel: Pass 2 (Finish der Survivors) ---------------------------
__global__ __launch_bounds__(256, 2)
void mandelbrotPass2Finish(
    uchar4* out, int* iterOut,
    const Survivor* surv, int survCount,
    int maxIter)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= survCount) return;

    Survivor s = surv[t];

    float zx = s.x, zy = s.y;
    int it = iterate_finish_loopcheck(s.cr, s.ci, s.it, maxIter, zx, zy);

    float3 col;
    if (it >= maxIter) {
        col = make_float3(0.0f,0.0f,0.0f);
    } else {
        col = otter::shade(it, maxIter, zx, zy, kPalette, kStripeF, kStripeAmp, kGamma);
    }

    out[s.idx] = make_uchar4(
        (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
        (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
        (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
        255);
    iterOut[s.idx] = it;
}

// ---------- ENTROPY & CONTRAST (unverändert) --------------------------------
__global__ void entropyKernel(
    const int* it, float* eOut,
    int w, int h, int tile, int maxIter)
{
    const bool doLog = Settings::debugLogging;
    int tX = blockIdx.x, tY = blockIdx.y;
    int startX = tX * tile, startY = tY * tile;

    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;
    int tileIndex = tY * tilesX + tX;

    if (doLog && threadIdx.x == 0) {
        char msg[256]; int n = 0;
        n = luchs::d_append_str(msg, sizeof(msg), n, "[ENTROPY-DEBUG] tX=");
        n = luchs::d_append_int(msg, sizeof(msg), n, tX);
        n = luchs::d_append_str(msg, sizeof(msg), n, " tY=");
        n = luchs::d_append_int(msg, sizeof(msg), n, tY);
        n = luchs::d_append_str(msg, sizeof(msg), n, " tile=");
        n = luchs::d_append_int(msg, sizeof(msg), n, tile);
        n = luchs::d_append_str(msg, sizeof(msg), n, " w=");
        n = luchs::d_append_int(msg, sizeof(msg), n, w);
        n = luchs::d_append_str(msg, sizeof(msg), n, " h=");
        n = luchs::d_append_int(msg, sizeof(msg), n, h);
        n = luchs::d_append_str(msg, sizeof(msg), n, " tilesX=");
        n = luchs::d_append_int(msg, sizeof(msg), n, tilesX);
        n = luchs::d_append_str(msg, sizeof(msg), n, " tilesY=");
        n = luchs::d_append_int(msg, sizeof(msg), n, tilesY);
        n = luchs::d_append_str(msg, sizeof(msg), n, " tileIndex=");
        n = luchs::d_append_int(msg, sizeof(msg), n, tileIndex);
        luchs::d_terminate(msg, sizeof(msg), n);
        LUCHS_LOG_DEVICE(msg);
    }

    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) histo[i] = 0;
    __syncthreads();

    const int total = tile * tile;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int dx = idx % tile, dy = idx / tile;
        int x = startX + dx, y = startY + dy;
        if (x >= w || y >= h) continue;
        int v = it[y * w + x];
        v = max(0, v);
        int bin = min(v * 256 / (maxIter + 1), 255);
        atomicAdd(&histo[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float entropy = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float p = float(histo[i]) / float(total);
            if (p > 0.0f) entropy -= p * __log2f(p);
        }
        eOut[tileIndex] = entropy;

        if (doLog) {
            char msg[128]; int n = 0;
            n = luchs::d_append_str(msg, sizeof(msg), n, "[ENTROPY] tile=(");
            n = luchs::d_append_int(msg, sizeof(msg), n, tX);
            n = luchs::d_append_str(msg, sizeof(msg), n, ",");
            n = luchs::d_append_int(msg, sizeof(msg), n, tY);
            n = luchs::d_append_str(msg, sizeof(msg), n, ") entropy=");
            n = luchs::d_append_float_fixed(msg, sizeof(msg), n, entropy, 5);
            luchs::d_terminate(msg, sizeof(msg), n);
            LUCHS_LOG_DEVICE(msg);
        }
    }
}

__global__ void contrastKernel(
    const float* e, float* cOut,
    int tilesX, int tilesY)
{
    const bool doLog = Settings::debugLogging;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty * tilesX + tx;
    float center = e[idx], sum = 0.0f;
    int cnt = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = tx + dx, ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            int nIdx = ny * tilesX + nx;
            sum += fabsf(e[nIdx] - center);
            ++cnt;
        }

    float contrast = (cnt > 0) ? sum / cnt : 0.0f;
    cOut[idx] = contrast;

    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        char msg[128]; int n = 0;
        n = luchs::d_append_str(msg, sizeof(msg), n, "[CONTRAST] tile=(");
        n = luchs::d_append_int(msg, sizeof(msg), n, tx);
        n = luchs::d_append_str(msg, sizeof(msg), n, ",");
        n = luchs::d_append_int(msg, sizeof(msg), n, ty);
        n = luchs::d_append_str(msg, sizeof(msg), n, ") contrast=");
        n = luchs::d_append_float_fixed(msg, sizeof(msg), n, contrast, 5);
        luchs::d_terminate(msg, sizeof(msg), n);
        LUCHS_LOG_DEVICE(msg);
    }
}

// ---------- Host: Entropy/Contrast Wrapper ----------------------------------
void computeCudaEntropyContrast(
    const int* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter)
{
    using clk = std::chrono::high_resolution_clock;
    auto start = clk::now();

    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;

    cudaMemset(d_e, 0, tilesX * tilesY * sizeof(float));

    entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
    cudaDeviceSynchronize();

    auto mid = clk::now();

    contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    cudaDeviceSynchronize();

    auto end = clk::now();

    if (Settings::performanceLogging) {
        double entropyMs = std::chrono::duration<double, std::milli>(mid - start).count();
        double contrastMs = std::chrono::duration<double, std::milli>(end - mid).count();
        LUCHS_LOG_HOST("[PERF] entropy=%.3f ms contrast=%.3f ms", entropyMs, contrastMs);
    } else if (Settings::debugLogging) {
        double entropyMs = std::chrono::duration<double, std::milli>(mid - start).count();
        double contrastMs = std::chrono::duration<double, std::milli>(end - mid).count();
        LUCHS_LOG_HOST("[TIME] Entropy %.3f ms | Contrast %.3f ms", entropyMs, contrastMs);
    }
}

// ---------- Host: Mandelbrot 2-Pass Wrapper ---------------------------------
namespace {
    // persistente temporäre Buffer (werden bei Bedarf vergrößert)
    Survivor* g_dSurvivors = nullptr;
    int*      g_dSurvCount = nullptr;
    size_t    g_survivorCap = 0;

    void ensureSurvivorCapacity(size_t need) {
        if (need <= g_survivorCap) return;
        if (g_dSurvivors) cudaFree(g_dSurvivors);
        if (g_dSurvCount) cudaFree(g_dSurvCount);
        cudaMalloc(&g_dSurvivors, need * sizeof(Survivor));
        cudaMalloc(&g_dSurvCount, sizeof(int));
        g_survivorCap = need;
    }
}

void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;

    // Block/Grids (etwas größer für bessere Occupancy, falls Regcount passt)
    dim3 block = Settings::performanceLogging ? dim3(32, 8) : dim3(16, 16);
    // Tipp: wenn ptxas Regcount moderat ist, probier 32x12
    // dim3 block = dim3(32,12);
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    // Survivor-Buffer (max. w*h)
    ensureSurvivorCapacity(size_t(w) * size_t(h));

    // Timing
    auto t0 = clk::now();
    auto t_launchStart = clk::now();

    // Pass 1
    cudaMemset(g_dSurvCount, 0, sizeof(int));
    mandelbrotPass1Warmup<<<grid, block>>>(out, d_it, g_dSurvivors, g_dSurvCount, w, h, zoom, offset, maxIter);

    // Survivor-Zahl holen
    int h_survCount = 0;
    cudaMemcpy(&h_survCount, g_dSurvCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Pass 2 (nur wenn nötig)
    if (h_survCount > 0) {
        int threads = 256;
        int blocks  = (h_survCount + threads - 1) / threads;
        mandelbrotPass2Finish<<<blocks, threads>>>(out, d_it, g_dSurvivors, h_survCount, maxIter);
    }

    auto t_launchEnd = clk::now();

    // Sync
    auto t_syncStart = clk::now();
    cudaDeviceSynchronize();
    auto t_syncEnd = clk::now();
    auto t1 = clk::now();

    // Logs im alten Format
    if (Settings::performanceLogging) {
        double launchMs = std::chrono::duration<double, std::milli>(t_launchEnd - t_launchStart).count();
        double syncMs   = std::chrono::duration<double, std::milli>(t_syncEnd - t_syncStart).count();
        double totalMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LUCHS_LOG_HOST("[PERF] mandelbrot: launch=%.3f ms sync=%.3f ms total=%.3f ms",
                       launchMs, syncMs, totalMs);
    } else if (Settings::debugLogging) {
        double launchMs = std::chrono::duration<double, std::milli>(t_launchEnd - t_launchStart).count();
        double syncMs   = std::chrono::duration<double, std::milli>(t_syncEnd - t_syncStart).count();
        double totalMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LUCHS_LOG_HOST("[TIME] Mandelbrot | Launch %.3f ms | Sync %.3f ms | Total %.3f ms",
                       launchMs, syncMs, totalMs);
    }
}
