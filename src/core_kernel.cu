///// MAUS: Fast P1 periodicity + adaptive slice sizing (Otter/Schneefuchs) — ASCII logs only
// core_kernel.cu — 2-Pass Mandelbrot (Warmup + Sliced Survivor Finish)
// 🐭 Maus: Kern schlank; deterministische ASCII-Logs.
// 🦦 Otter: Smooth Coloring, adaptive Warmup & Slice-Größe.
// 🦊 Schneefuchs: Warp-synchron, CHUNKed, Periodizitätsprobe in Pass 1/2, reduzierte Divergenz.

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
    constexpr int   WARP_CHUNK        = 64;     // vorher 32

    // Periodizitätsprobe (Pass 2 – Finish)
    constexpr int   LOOP_CHECK_EVERY  = 16;     // vorher 32
    constexpr float LOOP_EPS2         = 1e-6f;  // vorher 5e-8
    constexpr int   LOOP_REQ_HITS     = 1;      // vorher ~2

    // 🦦 Otter FAST-Preset: Pass-1 Periodizität schärfen
    constexpr int   P1_LOOP_EVERY     = 48;     // vorher 64 (dichter prüfen)
    constexpr float P1_LOOP_EPS2      = 2e-7f;  // vorher 1e-7 (toleranter)
    constexpr int   P1_LOOP_REQ_HITS  = 1;      // vorher 2  (schneller akzeptieren)

    // Basis-Warmup; kann via __constant__ adaptiv angehoben werden
    constexpr int   WARMUP_IT_BASE    = 1024;

    // Slice-Steps in Pass 2 (Startwert)
    constexpr int   FINISH_SLICE_IT   = 1024;
    constexpr int   MAX_SLICES        = 64;     // Sicherheitskappung

    // Otter: Paletten-/Shading-Defaults
    constexpr otter::Palette kPalette   = otter::Palette::Glacier; // Aurora/Glacier/Ember
    constexpr float          kStripeF   = 3.0f;
    constexpr float          kStripeAmp = 0.10f;
    constexpr float          kGamma     = 2.2f;
}

// ---------- Adaptive Warmup (device constant) --------------------------------
// Schneefuchs: keine Kernel-Signatur anpassen, via __constant__ injizieren.
__device__ __constant__ int d_warmup_it = WARMUP_IT_BASE;

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
    // Hauptkardioide
    float xm = x - 0.25f;
    float q  = xm * xm + y * y;
    if (q * (q + xm) <= 0.25f * y * y) return true;
    // period-2 Bulb um (-1,0), r=0.25
    float xp = x + 1.0f;
    if (xp * xp + y * y <= 0.0625f) return true;
    return false;
}

// ---------- Iteration (CHUNKed) ---------------------------------------------
// Pass 1: Warmup MIT leichter Periodizitätsprobe
__device__ __forceinline__ int iterate_warmup_noLoop(
    float cr, float ci, int maxSteps, float& x, float& y, bool& interiorFlag)
{
    x = 0.0f; y = 0.0f;
    int it = 0;
    interiorFlag = false;

    float px = x, py = y; // Referenz für P1-Loop-Check
    int   pc = 0;
    int   close_hits = 0;

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    bool active = true;

#pragma unroll 1
    for (int k = 0; k < maxSteps; k += WARP_CHUNK) {
#pragma unroll 1
        for (int s = 0; s < WARP_CHUNK; ++s) {
            if (!active) { ++pc; continue; }

            float xx = x * x;
            float yy = y * y;
            if (xx + yy > 4.0f) { active = false; ++pc; continue; }

            // z = z^2 + c (mit FMA)
            float xt = fmaf(x, x, -yy) + cr;   // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);         // 2*x*y + ci
            x = xt;
            ++it; ++pc;

            // leichte Periodizitätsprobe (FAST-Preset)
            if (pc >= P1_LOOP_EVERY) {
                float dx = x - px, dy = y - py;
                float d2 = dx*dx + dy*dy;
                if (d2 < P1_LOOP_EPS2) {
                    if (++close_hits >= P1_LOOP_REQ_HITS) {
                        active        = false;
                        interiorFlag  = true;   // im Kernel konsumiert
                        it            = maxSteps; // markiere "innen"
                    }
                } else {
                    close_hits = 0;
                }
                px = x; py = y; pc = 0;
            }

            if (it >= maxSteps) { active = false; break; }
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
    int   it;      // bisherige Iterationen (WARMUP_IT / Slice-Zwischenstand)
    int   idx;     // Pixelindex
};

// ---------- Pass-2 Slice Iteration ------------------------------------------
struct SliceResult { int it; float x, y; bool escaped; bool interior; };

// bis zu 'sliceSteps' Finish-Schritte inkl. Periodizitätsprobe.
__device__ __forceinline__ SliceResult iterate_finish_slice(
    float cr, float ci, int start_it, int maxIter,
    float x, float y, int sliceSteps)
{
    // Schneefuchs: analytischer Early-Exit
    if (insideMainCardioidOrBulb(cr, ci)) {
        return { maxIter, x, y, /*escaped*/false, /*interior*/true };
    }

    int it = start_it;

    float px = x, py = y;    // Referenz für Loop-Check
    int   pc = 0;
    int   close_hits = 0;

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    bool active = true;
    bool escaped = false;
    bool interior = false;

#pragma unroll 1
    for (int k = 0; k < sliceSteps; k += WARP_CHUNK) {
#pragma unroll 1
        for (int s = 0; s < WARP_CHUNK; ++s) {
            if (!active) { ++pc; continue; }

            float x2 = x * x;
            float y2 = y * y;
            if (x2 + y2 > 4.0f) { active = false; escaped = true; ++pc; continue; }

            float xt = fmaf(x, x, -y2) + cr;   // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);         // 2*x*y + ci
            x = xt;
            ++it; ++pc;

            if (pc >= LOOP_CHECK_EVERY) {
                float dx = x - px, dy = y - py;
                float d2 = dx*dx + dy*dy;
                if (d2 < LOOP_EPS2) {
                    if (++close_hits >= LOOP_REQ_HITS) {
                        active   = false;
                        interior = true;
                        it = maxIter; // „innen“ markieren
                    }
                } else {
                    close_hits = 0;
                }
                px = x; py = y; pc = 0;
            }
            if (it >= maxIter) { active = false; break; }
        }
        unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }
    return { it, x, y, escaped, interior };
}

// ---------- Kernel: Pass 1 (Warmup + Kompaktierung) -------------------------
__global__ __launch_bounds__(256, 2)
void mandelbrotPass1Warmup(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    Survivor* __restrict__ surv, int* __restrict__ survCount,
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

    // Otter: Innenpunkte *ohne* Iterationsschleife behandeln
    if (insideMainCardioidOrBulb(c.x, c.y)) {
        out[idx]     = make_uchar4(0,0,0,255);
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

    // Warmup bis d_warmup_it (adaptiv vom Host gesetzt)
    const int warmupSteps = d_warmup_it;

    float zx=0.0f, zy=0.0f;
    bool interior = false;
    int itWarm = iterate_warmup_noLoop(c.x, c.y, warmupSteps, zx, zy, interior);

    if (interior) {
        out[idx]     = make_uchar4(0,0,0,255);
        iterOut[idx] = maxIter;
        return;
    }

    const float norm = zx*zx + zy*zy;
    const bool escaped = (itWarm < warmupSteps) && (norm > 4.0f);

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

    // Survivor → warp-aggregated kompakter Push (robust für 2D-Blocks)
    unsigned actMask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    actMask = __activemask();
#endif
    const bool isSurvivor = true;
    const unsigned ballot = __ballot_sync(actMask, isSurvivor);
    const int      voteCount = __popc(ballot);

    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane       = (linear_tid & 31);
    const unsigned laneMask = ballot & ((1u << lane) - 1u);
    const int      prefix   = __popc(laneMask);

    int base = 0;
    const int leader = __ffs(ballot) - 1;                 // erste aktive Lane
    if (lane == leader) {
        base = atomicAdd(survCount, voteCount);
    }
    base = __shfl_sync(ballot, base, leader);             // nur über Survivors shufflen

    // schreiben
    Survivor s;
    s.x = zx; s.y = zy; s.cr = c.x; s.ci = c.y; s.it = itWarm; s.idx = idx;
    surv[base + prefix] = s;

    // (kein Ausgabeschreiben hier—Pass 2 kümmert sich)
}

// ---------- Kernel: Pass 2 (Slice + Re-Kompaktierung) -----------------------
// Otter/Schneefuchs: sliceIt als Parameter (intern), Header unangetastet
__global__ __launch_bounds__(128, 2)
void mandelbrotPass2Slice(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    const Survivor* __restrict__ survIn, int survInCount,
    Survivor* __restrict__ survOut, int* __restrict__ survOutCount,
    int maxIter, int sliceIt)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= survInCount) return;

    Survivor s = survIn[t];

    // Guard: falls Survivor dennoch innen war (Randfälle)
    if (insideMainCardioidOrBulb(s.cr, s.ci)) {
        out[s.idx]     = make_uchar4(0,0,0,255);
        iterOut[s.idx] = maxIter;
        return;
    }

    SliceResult r = iterate_finish_slice(s.cr, s.ci, s.it, maxIter, s.x, s.y, sliceIt);

    if (r.escaped) {
        float3 col = otter::shade(r.it, maxIter, r.x, r.y, kPalette, kStripeF, kStripeAmp, kGamma);
        out[s.idx] = make_uchar4(
            (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
            255);
        iterOut[s.idx] = r.it;
        return;
    }

    if (r.it >= maxIter || r.interior) {
        // innen/schwarz
        out[s.idx]     = make_uchar4(0,0,0,255);
        iterOut[s.idx] = r.it;
        return;
    }

    // lebt weiter → kompakt in 'survOut' pushen (warp-aggregated)
    unsigned actMask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    actMask = __activemask();
#endif
    const bool isSurvivor = true;
    const unsigned ballot = __ballot_sync(actMask, isSurvivor);
    const int      voteCount = __popc(ballot);

    const int lane = threadIdx.x & 31;
    const unsigned laneMask = ballot & ((1u << lane) - 1u);
    const int prefix = __popc(laneMask);

    int base = 0;
    const int leader = __ffs(ballot) - 1;
    if (lane == leader) {
        base = atomicAdd(survOutCount, voteCount);
    }
    base = __shfl_sync(ballot, base, leader);

    Survivor ns;
    ns.x = r.x; ns.y = r.y; ns.cr = s.cr; ns.ci = s.ci; ns.it = r.it; ns.idx = s.idx;
    survOut[base + prefix] = ns;
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

// ---------- Host: Mandelbrot 2-Pass Wrapper (Sliced Finish) -----------------
namespace {
    // zwei Survivor-Puffer + Zähler (Ping-Pong), persistent & resizable
    Survivor* g_dSurvivorsA = nullptr;
    Survivor* g_dSurvivorsB = nullptr;
    int*      g_dSurvCountA = nullptr;
    int*      g_dSurvCountB = nullptr;
    size_t    g_survivorCap = 0;

    // Merker für adaptive Warmup-Steuerung
    double    g_prevSurvivorsPct = -1.0;

    void ensureSurvivorCapacity(size_t need) {
        if (need <= g_survivorCap) return;
        if (g_dSurvivorsA) cudaFree(g_dSurvivorsA);
        if (g_dSurvivorsB) cudaFree(g_dSurvivorsB);
        if (g_dSurvCountA) cudaFree(g_dSurvCountA);
        if (g_dSurvCountB) cudaFree(g_dSurvCountB);
        cudaMalloc(&g_dSurvivorsA, need * sizeof(Survivor));
        cudaMalloc(&g_dSurvivorsB, need * sizeof(Survivor));
        cudaMalloc(&g_dSurvCountA, sizeof(int));
        cudaMalloc(&g_dSurvCountB, sizeof(int));
        g_survivorCap = need;
    }

    inline int chooseWarmupIt(int maxIter) {
        // Otter: einfache Heuristik basierend auf letzter Survivors-Quote
        int warm = WARMUP_IT_BASE;
        if (g_prevSurvivorsPct >= 90.0)      warm = std::min(maxIter, WARMUP_IT_BASE * 3);
        else if (g_prevSurvivorsPct >= 80.0) warm = std::min(maxIter, WARMUP_IT_BASE * 2);
        else if (g_prevSurvivorsPct >= 60.0) warm = std::min(maxIter, (WARMUP_IT_BASE * 3) / 2);
        return warm;
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
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    // Survivor-Buffer (max. w*h)
    ensureSurvivorCapacity(size_t(w) * size_t(h));

    // Adaptive Warmup-Schritte in Device-Const schreiben
    const int warmupIt = chooseWarmupIt(maxIter);
    cudaMemcpyToSymbol(d_warmup_it, &warmupIt, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] warmup_it=%d prev_survivors=%.2f%%", warmupIt, g_prevSurvivorsPct);
    }

    auto t0 = clk::now();

    // Pass 1
    cudaMemset(g_dSurvCountA, 0, sizeof(int));
    mandelbrotPass1Warmup<<<grid, block>>>(out, d_it, g_dSurvivorsA, g_dSurvCountA, w, h, zoom, offset, maxIter);

    // Survivor-Zahl holen + Log wie gehabt
    int h_survA = 0;
    cudaMemcpy(&h_survA, g_dSurvCountA, sizeof(int), cudaMemcpyDeviceToHost);
    const double survPct = (double)h_survA * 100.0 / (double(w) * double(h));
    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] survivors=%d (%.2f%% of %d)", h_survA, survPct, w*h);
    }
    // Merken für nächstes Frame
    g_prevSurvivorsPct = survPct;

    // Pass 2: Slices mit Re-Kompaktierung (adaptives Slice-Sizing)
    if (h_survA > 0) {
        cudaFuncSetCacheConfig(mandelbrotPass2Slice, cudaFuncCachePreferL1);

        int threads = 128;
        int slice   = 0;
        int sliceIt = FINISH_SLICE_IT; // Startwert, wird dynamisch geboostet

        Survivor* curBuf = g_dSurvivorsA;
        Survivor* nxtBuf = g_dSurvivorsB;
        int*      curCnt = g_dSurvCountA;
        int*      nxtCnt = g_dSurvCountB;
        int       h_cur  = h_survA;

        while (h_cur > 0 && slice < MAX_SLICES) {
            cudaMemset(nxtCnt, 0, sizeof(int));
            int blocks  = (h_cur + threads - 1) / threads;

            mandelbrotPass2Slice<<<blocks, threads>>>(
                out, d_it, curBuf, h_cur, nxtBuf, nxtCnt, maxIter, sliceIt);

            int h_next = 0;
            cudaMemcpy(&h_next, nxtCnt, sizeof(int), cudaMemcpyDeviceToHost);

            if (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[PERF] slice=%d steps=%d survivors_in=%d survivors_out=%d",
                               slice, sliceIt, h_cur, h_next);
            }

            // 🦦 Otter: adaptive Slice-Vergrößerung bei geringem Fortschritt
            const int drop = h_cur - h_next;
            const double dropPct = (h_cur > 0) ? (double)drop / (double)h_cur : 1.0;
            if (dropPct < 0.003 && sliceIt < (maxIter / 2)) {
                sliceIt = min(sliceIt * 2, maxIter / 2);
                if (Settings::performanceLogging) {
                    LUCHS_LOG_HOST("[PERF] adapt_slice_it=%d (dropPct=%.4f)", sliceIt, dropPct);
                }
            }

            // Ping-Pong
            std::swap(curBuf, nxtBuf);
            std::swap(curCnt, nxtCnt);
            h_cur = h_next;
            ++slice;
        }
    }

    // Sync & Zeit
    cudaDeviceSynchronize();
    auto t1 = clk::now();

    if (Settings::performanceLogging) {
        double totalMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LUCHS_LOG_HOST("[PERF] mandelbrot (hybrid-sliced): total=%.3f ms", totalMs);
    } else if (Settings::debugLogging) {
        double totalMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LUCHS_LOG_HOST("[TIME] Mandelbrot Sliced | Total %.3f ms", totalMs);
    }
}
