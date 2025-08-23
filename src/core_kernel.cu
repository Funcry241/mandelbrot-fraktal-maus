///// MAUS: Fast P1 periodicity + adaptive slice sizing + frame-budget pacing (Otter/Schneefuchs) — ASCII-only
// core_kernel.cu — 2-Pass Mandelbrot (Warmup + Sliced Survivor Finish)
// + Metric AA (DE-based): |dz/dc|-Distance-Estimator smooths edges without supersampling / higher precision.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "common.hpp"
#include "luchs_device_format.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"
#include "otter_color.hpp"

// ---------- Tuning -----------------------------------------------------------
namespace {
    // Larger CHUNK -> fewer ballots, same image quality.
    constexpr int   WARP_CHUNK        = 64;     // was 32
    static_assert((WARP_CHUNK % 32) == 0, "WARP_CHUNK must be a multiple of 32 (warp size)");

    // Periodicity check (Pass 2 - Finish)
    constexpr int   LOOP_CHECK_EVERY  = 16;     // was 32
    constexpr float LOOP_EPS2         = 1e-6f;  // was 5e-8
    constexpr int   LOOP_REQ_HITS     = 1;      // was ~2

    // Otter FAST-preset: sharpen Pass-1 periodicity
    constexpr int   P1_LOOP_EVERY     = 48;     // was 64 (denser)
    constexpr float P1_LOOP_EPS2      = 2e-7f;  // was 1e-7 (more tolerant)
    constexpr int   P1_LOOP_REQ_HITS  = 1;      // was 2  (faster accept)

    // Base warmup; can be raised via __constant__ adaptively
    constexpr int   WARMUP_IT_BASE    = 1024;

    // Slice steps in Pass 2 (start)
    constexpr int   FINISH_SLICE_IT   = 1024;
    constexpr int   MAX_SLICES        = 64;     // safety cap

    // Otter: palette / shading defaults
    constexpr otter::Palette kPalette   = otter::Palette::Glacier; // Aurora/Glacier/Ember
    constexpr float          kStripeF   = 3.0f;
    constexpr float          kStripeAmp = 0.10f;

    // Gamma: fallback independent from Settings::EnableSRGB (older branches may not have SRGB FBO)
    #if defined(OTTER_FRAMEBUFFER_SRGB) && (OTTER_FRAMEBUFFER_SRGB)
        constexpr float      kGamma     = 1.0f;   // FBO converts -> no extra gamma here
    #else
        constexpr float      kGamma     = 2.2f;   // linear RGBA8 pipeline
    #endif

    // Frame-budget pacing (host side, log-independent; no events required)
    constexpr double FRAME_BUDGET_FALLBACK_MS = 16.6667; // if no FPS cap
    constexpr double KERNEL_BUDGET_FRACTION   = 0.62;    // share for fractal path
    constexpr double MIN_BUDGET_MS            = 6.0;     // hard lower bound

    // Slice adaptation (EMA over survivor-drop ratio)
    constexpr float DROP_EMA_ALPHA            = 0.25f;   // reaction speed
    constexpr float DROP_UPPER_BACKOFF        = 0.30f;   // >30%: halve slice
    constexpr float DROP_LOWER_ACCEL          = 0.005f;  // <0.5%: double slice

    // "Deep close" interior detection (in addition to normal periodicity check)
    constexpr float DEEP_EPS2                 = 1e-10f;  // very close return
    constexpr int   DEEP_REQ_HITS             = 3;       // require multiple hits

    // Metric-AA strength (bigger = softer edge, smaller = sharper)
    constexpr float AA_K                      = 2.0f;    // 1.5 .. 3.0 recommended
}

// ---------- Adaptive Warmup (device constant) --------------------------------
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
    float xm = x - 0.25f;
    float q  = xm * xm + y * y;
    if (q * (q + xm) <= 0.25f * y * y) return true;
    float xp = x + 1.0f;
    if (xp * xp + y * y <= 0.0625f) return true;
    return false;
}

__device__ __forceinline__ float smooth01(float x) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// ---------- Iteration (CHUNKed) ---------------------------------------------
// Pass 1: Warmup WITH light periodicity test + dz/dc accumulation
__device__ __forceinline__ int iterate_warmup_noLoop(
    float cr, float ci, int maxSteps,
    float& x, float& y, bool& interiorFlag,
    float& dx, float& dy) // dz/dc
{
    x = 0.0f; y = 0.0f; dx = 0.0f; dy = 0.0f;
    int it = 0;
    interiorFlag = false;

    float px = x, py = y; // reference for P1 loop check
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

            // dz/dc <- 2*z*dz/dc + 1
            float ndx = 2.0f * (x * dx - y * dy) + 1.0f;
            float ndy = 2.0f * (x * dy + y * dx);

            float xt = fmaf(x, x, -yy) + cr;   // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);         // 2*x*y + ci
            x = xt;
            dx = ndx; dy = ndy;

            ++it; ++pc;

            if (pc >= P1_LOOP_EVERY) {
                float ex = x - px, ey = y - py;
                float d2 = ex*ex + ey*ey;
                if (d2 < P1_LOOP_EPS2) {
                    if (++close_hits >= P1_LOOP_REQ_HITS) {
                        active        = false;
                        interiorFlag  = true;
                        it            = maxSteps;
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
struct Survivor { float x, y, dx, dy, cr, ci; int it, idx; };

// ---------- Pass-2 Slice Iteration ------------------------------------------
struct SliceResult { int it; float x, y, dx, dy; bool escaped; bool interior; float de; };

__device__ __forceinline__ SliceResult iterate_finish_slice(
    float cr, float ci, int start_it, int maxIter,
    float x, float y, float dx, float dy,
    int sliceSteps)
{
    if (insideMainCardioidOrBulb(cr, ci)) {
        return { maxIter, x, y, dx, dy, false, true, 0.0f };
    }

    int it = start_it;
    float px = x, py = y;
    int   pc = 0, close_hits = 0, deep_hits = 0;

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    bool active = true, escaped = false, interior = false;
    float deOut = 0.0f;

#pragma unroll 1
    for (int k = 0; k < sliceSteps; k += WARP_CHUNK) {
#pragma unroll 1
        for (int s = 0; s < WARP_CHUNK; ++s) {
            if (!active) { ++pc; continue; }

            float x2 = x * x, y2 = y * y;
            if (x2 + y2 > 4.0f) {
                // Distance estimator on escape
                float r   = sqrtf(x2 + y2);
                float dd  = fmaxf(1e-30f, sqrtf(dx*dx + dy*dy));
                deOut     = (r > 0.0f) ? (r * logf(r) / dd) : 0.0f;
                active    = false; escaped = true; ++pc; continue;
            }

            // dz/dc <- 2*z*dz/dc + 1
            float ndx = 2.0f * (x * dx - y * dy) + 1.0f;
            float ndy = 2.0f * (x * dy + y * dx);
            dx = ndx; dy = ndy;

            float xt = fmaf(x, x, -y2) + cr;
            y = fmaf(2.0f * x, y, ci);
            x = xt;
            ++it; ++pc;

            if (pc >= LOOP_CHECK_EVERY) {
                float ex = x - px, ey = y - py;
                float d2 = ex*ex + ey*ey;
                if (d2 < LOOP_EPS2) {
                    if (++close_hits >= LOOP_REQ_HITS) { active = false; interior = true; it = maxIter; }
                } else {
                    close_hits = 0;
                }
                if (d2 < DEEP_EPS2 && (x2 + y2) < 4.0f) {
                    if (++deep_hits >= DEEP_REQ_HITS) { active = false; interior = true; it = maxIter; }
                } else {
                    deep_hits = 0;
                }
                px = x; py = y; pc = 0;
            }
            if (it >= maxIter) { active = false; break; }
        }
        unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }
    return { it, x, y, dx, dy, escaped, interior, deOut };
}

// ---------- Kernel: Pass 1 (Warmup + compaction) ----------------------------
__global__ __launch_bounds__(256, 2)
void mandelbrotPass1Warmup(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    Survivor* __restrict__ surv, int* __restrict__ survCount,
    int w, int h, float zoom, float2 offset,
    int maxIter, float pixR) // pixel footprint radius in c-plane
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

    const int warmupSteps = d_warmup_it;

    float zx=0.0f, zy=0.0f, dx=0.0f, dy=0.0f; bool interior = false;
    int itWarm = iterate_warmup_noLoop(c.x, c.y, warmupSteps, zx, zy, interior, dx, dy);

    if (interior) {
        out[idx]     = make_uchar4(0,0,0,255);
        iterOut[idx] = maxIter;
        return;
    }

    const float r2 = zx*zx + zy*zy;
    const bool escaped = (itWarm < warmupSteps) && (r2 > 4.0f);

    if (escaped) {
        // Metric-AA: coverage via distance estimator vs pixel radius
        const float r  = sqrtf(r2);
        const float dd = fmaxf(1e-30f, sqrtf(dx*dx + dy*dy));
        const float de = (r > 0.0f) ? (r * logf(r) / dd) : 0.0f;
        const float cov = smooth01(de / (AA_K * pixR));

        float3 col = otter::shade(itWarm, maxIter, zx, zy, kPalette, kStripeF, kStripeAmp, kGamma);
        col.x *= cov; col.y *= cov; col.z *= cov;

        out[idx] = make_uchar4(
            (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
            255);
        iterOut[idx] = itWarm;
        return;
    }

    // Survivors: black immediately (prevent ghosting)
    out[idx] = make_uchar4(0,0,0,255);

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
    const int leader = __ffs(ballot) - 1;
    if (lane == leader) base = atomicAdd(survCount, voteCount);
    base = __shfl_sync(ballot, base, leader);

    Survivor s; s.x = zx; s.y = zy; s.dx = dx; s.dy = dy; s.cr = c.x; s.ci = c.y; s.it = itWarm; s.idx = idx;
    surv[base + prefix] = s;
}

// ---------- Kernel: Pass 2 (Slice + compaction) -----------------------------
__global__ __launch_bounds__(128, 2)
void mandelbrotPass2Slice(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    const Survivor* __restrict__ survIn, int survInCount,
    Survivor* __restrict__ survOut, int* __restrict__ survOutCount,
    int maxIter, int sliceIt, float pixR) // pixel footprint radius
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= survInCount) return;

    Survivor s = survIn[t];

    if (insideMainCardioidOrBulb(s.cr, s.ci)) {
        out[s.idx]     = make_uchar4(0,0,0,255);
        iterOut[s.idx] = maxIter;
        return;
    }

    SliceResult r = iterate_finish_slice(s.cr, s.ci, s.it, maxIter, s.x, s.y, s.dx, s.dy, sliceIt);

    if (r.escaped) {
        const float cov = smooth01(r.de / (AA_K * pixR));
        float3 col = otter::shade(r.it, maxIter, r.x, r.y, kPalette, kStripeF, kStripeAmp, kGamma);
        col.x *= cov; col.y *= cov; col.z *= cov;

        out[s.idx] = make_uchar4(
            (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
            255);
        iterOut[s.idx] = r.it;
        return;
    }

    if (r.it >= maxIter || r.interior) {
        out[s.idx]     = make_uchar4(0,0,0,255);
        iterOut[s.idx] = r.it;
        return;
    }

    // Not finished yet -> black immediately (prevent ghosting)
    out[s.idx] = make_uchar4(0,0,0,255);

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
    if (lane == leader) base = atomicAdd(survOutCount, voteCount);
    base = __shfl_sync(ballot, base, leader);

    Survivor ns; ns.x = r.x; ns.y = r.y; ns.dx = r.dx; ns.dy = r.dy; ns.cr = s.cr; ns.ci = s.ci; ns.it = r.it; ns.idx = s.idx;
    survOut[base + prefix] = ns;
}

// ---------- ENTROPY & CONTRAST (Kernels) ------------------------------------
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
    const bool doLog = (Settings::performanceLogging || Settings::debugLogging);

    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;

    cudaMemset(d_e, 0, tilesX * tilesY * sizeof(float));

    // Event-based timing only for logging; no functional impact.
    if (doLog) {
        cudaEvent_t evStart, evMid, evEnd;
        cudaEventCreate(&evStart); cudaEventCreate(&evMid); cudaEventCreate(&evEnd);

        cudaEventRecord(evStart, 0);
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        cudaEventRecord(evMid, 0);

        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
        cudaEventRecord(evEnd, 0);
        cudaEventSynchronize(evEnd); // timing only

        float entropyMs=0.f, contrastMs=0.f;
        cudaEventElapsedTime(&entropyMs, evStart, evMid);
        cudaEventElapsedTime(&contrastMs, evMid, evEnd);

        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] entropy=%.3f ms contrast=%.3f ms", entropyMs, contrastMs);
        } else if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[TIME] Entropy %.3f ms | Contrast %.3f ms", entropyMs, contrastMs);
        }

        cudaEventDestroy(evStart); cudaEventDestroy(evMid); cudaEventDestroy(evEnd);
    } else {
        // No events -> launch directly; no extra syncs.
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    }
}

// ---------- Host: Mandelbrot 2-Pass Wrapper (Sliced Finish) -----------------
namespace {
    Survivor* g_dSurvivorsA = nullptr;
    Survivor* g_dSurvivorsB = nullptr;
    int*      g_dSurvCountA = nullptr;
    int*      g_dSurvCountB = nullptr;
    size_t    g_survivorCap = 0;

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
        int warm = WARMUP_IT_BASE;
        if (g_prevSurvivorsPct >= 90.0)      warm = std::min(maxIter, WARMUP_IT_BASE * 3);
        else if (g_prevSurvivorsPct >= 80.0) warm = std::min(maxIter, WARMUP_IT_BASE * 2);
        else if (g_prevSurvivorsPct >= 60.0) warm = std::min(maxIter, (WARMUP_IT_BASE * 3) / 2);
        // Cap to maxIter/3 to avoid warmup spikes
        warm = std::min(warm, std::max(64, maxIter / 3));
        return warm;
    }

    inline double frameBudgetMsFromSettings() {
        if (Settings::capFramerate && Settings::capTargetFps > 0) {
            return std::max(MIN_BUDGET_MS, 1000.0 / double(Settings::capTargetFps));
        }
        return std::max(MIN_BUDGET_MS, FRAME_BUDGET_FALLBACK_MS);
    }
}

void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;
    const bool doLog = (Settings::performanceLogging || Settings::debugLogging);

    dim3 block(32, 8); // 256 threads
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    ensureSurvivorCapacity(size_t(w) * size_t(h));

    const int warmupIt = chooseWarmupIt(maxIter);
    cudaMemcpyToSymbol(d_warmup_it, &warmupIt, sizeof(int), 0, cudaMemcpyHostToDevice);

    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] warmup_it=%d prev_survivors=%.2f%%", warmupIt, g_prevSurvivorsPct);
    }

    // Prefer L1
    cudaFuncSetCacheConfig(mandelbrotPass1Warmup, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mandelbrotPass2Slice,  cudaFuncCachePreferL1);

    // Pixel footprint in c-plane (half pixel diagonal), robust clamp
    const float scale = 1.0f / zoom;
    const float spanX = 3.5f * scale;
    const float spanY = spanX * (float)h / (float)w;
    const float pixDiag = std::sqrt((spanX / w) * (spanX / w) + (spanY / h) * (spanY / h));
    const float pixR  = fmaxf(1e-25f, 0.5f * pixDiag);

    // Budget (log-independent); host timing + implicit sync via counter memcpy
    const double frameBudgetMs  = frameBudgetMsFromSettings();
    const double kernelBudgetMs = frameBudgetMs * KERNEL_BUDGET_FRACTION;
    const auto   hostStart      = clk::now();

    // Pass 1
    cudaMemset(g_dSurvCountA, 0, sizeof(int));
    mandelbrotPass1Warmup<<<grid, block>>>(out, d_it, g_dSurvivorsA, g_dSurvCountA, w, h, zoom, offset, maxIter, pixR);

    int h_survA = 0;
    cudaMemcpy(&h_survA, g_dSurvCountA, sizeof(int), cudaMemcpyDeviceToHost); // waits for P1
    const double survPct = (double)h_survA * 100.0 / (double(w) * double(h));
    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] survivors=%d (%.2f%% of %d)", h_survA, survPct, w*h);
    }
    g_prevSurvivorsPct = survPct;

    // P1 time (host)
    double p1Ms = std::chrono::duration<double, std::milli>(clk::now() - hostStart).count();
    if (h_survA <= 0) {
        if (doLog) {
            if (Settings::performanceLogging) LUCHS_LOG_HOST("[PERF] mandelbrot (hybrid-sliced): total=%.3f ms", p1Ms);
            else if (Settings::debugLogging)   LUCHS_LOG_HOST("[TIME] Mandelbrot Sliced | Total %.3f ms", p1Ms);
        }
        return;
    }
    if (p1Ms > kernelBudgetMs && doLog) {
        LUCHS_LOG_HOST("[PERF] budget_hit after P1: p1=%.3f ms budget=%.3f ms -> defer P2", p1Ms, kernelBudgetMs);
    }

    // Pass 2 (sliced) — budget control via host timing
    int threads = 128;
    int slice   = 0;
    int sliceIt = FINISH_SLICE_IT;

    Survivor* curBuf = g_dSurvivorsA;
    Survivor* nxtBuf = g_dSurvivorsB;
    int*      curCnt = g_dSurvCountA;
    int*      nxtCnt = g_dSurvCountB;
    int       h_cur  = h_survA;

    float emaDrop = 0.2f; // moderate start

    while (h_cur > 0 && slice < MAX_SLICES) {
        // Budget check before slice (P1 already included)
        double elapsedMs = std::chrono::duration<double, std::milli>(clk::now() - hostStart).count();
        if (elapsedMs >= kernelBudgetMs) {
            if (doLog) LUCHS_LOG_HOST("[PERF] budget_exhausted before slice %d: elapsed=%.3f ms budget=%.3f ms", slice, elapsedMs, kernelBudgetMs);
            break;
        }

        cudaMemset(nxtCnt, 0, sizeof(int));
        int blocks  = (h_cur + threads - 1) / threads;

        mandelbrotPass2Slice<<<blocks, threads>>>(
            out, d_it, curBuf, h_cur, nxtBuf, nxtCnt, maxIter, sliceIt, pixR);

        // Read counter (waits for slice kernel)
        int h_next = 0;
        cudaMemcpy(&h_next, nxtCnt, sizeof(int), cudaMemcpyDeviceToHost);

        elapsedMs = std::chrono::duration<double, std::milli>(clk::now() - hostStart).count();
        if (doLog) {
            LUCHS_LOG_HOST("[PERF] slice=%d steps=%d survivors_in=%d survivors_out=%d elapsed=%.3f ms (budget=%.3f)",
                           slice, sliceIt, h_cur, h_next, elapsedMs, kernelBudgetMs);
        }
        if (elapsedMs >= kernelBudgetMs) {
            if (doLog) LUCHS_LOG_HOST("[PERF] budget_stop at slice %d", slice);
            std::swap(curBuf, nxtBuf); std::swap(curCnt, nxtCnt); h_cur = h_next; ++slice;
            break; // rest next frame
        }

        const int drop = h_cur - h_next;
        const float dropPct = (h_cur > 0) ? float(drop) / float(h_cur) : 1.0f;
        emaDrop = (1.0f - DROP_EMA_ALPHA) * emaDrop + DROP_EMA_ALPHA * dropPct;

        // Adapt slice length (hysteresis)
        if (emaDrop < DROP_LOWER_ACCEL && sliceIt < (maxIter / 2)) {
            sliceIt = std::min(sliceIt * 2, maxIter / 2);
            if (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[PERF] adapt_slice_it=%d (emaDrop=%.4f)", sliceIt, emaDrop);
            }
        } else if (emaDrop > DROP_UPPER_BACKOFF && sliceIt > FINISH_SLICE_IT) {
            sliceIt = std::max(sliceIt / 2, FINISH_SLICE_IT);
            if (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[PERF] backoff_slice_it=%d (emaDrop=%.4f)", sliceIt, emaDrop);
            }
        }

        std::swap(curBuf, nxtBuf);
        std::swap(curCnt, nxtCnt);
        h_cur = h_next;
        ++slice;
    }

    // Final log (host total)
    if (doLog) {
        double totalMs = std::chrono::duration<double, std::milli>(clk::now() - hostStart).count();
        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] mandelbrot (hybrid-sliced): total=%.3f ms (budget=%.3f ms)", totalMs, kernelBudgetMs);
        } else if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[TIME] Mandelbrot Sliced | Total %.3f ms", totalMs);
        }
    }
}
