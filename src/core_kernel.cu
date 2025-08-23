// 2-Pass Mandelbrot (Warmup + Sliced Finish) + Metric-AA + Eye-Candy (hue/glow/orbit)
// ASCII-only, NVCC clean. Uses otter::shade() for base coloring.

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
    constexpr int   WARP_CHUNK        = 64;     // bigger chunk reduces ballots
    static_assert((WARP_CHUNK % 32) == 0, "WARP_CHUNK must be a multiple of 32");

    // Periodicity probes
    constexpr int   LOOP_CHECK_EVERY  = 16;
    constexpr float LOOP_EPS2         = 1e-6f;
    constexpr int   LOOP_REQ_HITS     = 1;

    constexpr int   P1_LOOP_EVERY     = 48;
    constexpr float P1_LOOP_EPS2      = 2e-7f;
    constexpr int   P1_LOOP_REQ_HITS  = 1;

    // Iter budgets
    constexpr int   WARMUP_IT_BASE    = 1024;
    constexpr int   FINISH_SLICE_IT   = 1024;
    constexpr int   MAX_SLICES        = 64;

    // Base palette/shading
    constexpr otter::Palette kPalette   = otter::Palette::Glacier;
    constexpr float          kStripeF   = 3.0f;
    constexpr float          kStripeAmp = 0.10f;

    #if defined(OTTER_FRAMEBUFFER_SRGB) && (OTTER_FRAMEBUFFER_SRGB)
        constexpr float      kGamma     = 1.0f;   // FBO handles gamma
    #else
        constexpr float      kGamma     = 2.2f;   // linear RGBA8 pipeline
    #endif

    // Frame budget pacing (host-side)
    constexpr double FRAME_BUDGET_FALLBACK_MS = 16.6667;
    constexpr double KERNEL_BUDGET_FRACTION   = 0.62;
    constexpr double MIN_BUDGET_MS            = 6.0;

    // Slice adaptation
    constexpr float DROP_EMA_ALPHA            = 0.25f;
    constexpr float DROP_UPPER_BACKOFF        = 0.30f;  // halve slice if >30% remain
    constexpr float DROP_LOWER_ACCEL          = 0.005f; // double slice if <0.5% remain

    // Deep-close interior helper
    constexpr float DEEP_EPS2                 = 1e-10f;
    constexpr int   DEEP_REQ_HITS             = 3;

    // Metric-AA softness (bigger = softer)
    constexpr float AA_K                      = 2.0f;

    // Orbit trap center (classic)
    constexpr float ORBIT_CX                  = -0.745f;
    constexpr float ORBIT_CY                  =  0.186f;
}

// ---------- Device constants: runtime knobs ---------------------------------
__device__ __constant__ int   d_warmup_it = WARMUP_IT_BASE;

__device__ __constant__ float  d_phase   = 0.0f;   // hue rotation phase (radians)
__device__ __constant__ float  d_glow    = 0.15f;  // edge glow strength (0..1)
__device__ __constant__ float  d_orbitK  = 6.0f;   // orbit trap falloff
__device__ __constant__ float  d_tintMix = 0.20f;  // max tint mix
__device__ __constant__ float3 d_tint    = {0.60f, 0.20f, 0.80f}; // purple-ish

// ---------- Helpers ----------------------------------------------------------
__device__ __forceinline__ float2 pixelToComplex(
    float px, float py, int w, int h, float spanX, float spanY, float2 offset)
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

// YIQ hue rotation (stable, low cost)
__device__ __forceinline__ float3 hueRotateYIQ(float3 rgb, float ang) {
    const float Y = 0.299f*rgb.x + 0.587f*rgb.y + 0.114f*rgb.z;
    float I =  0.596f*rgb.x - 0.274f*rgb.y - 0.322f*rgb.z;
    float Q =  0.211f*rgb.x - 0.523f*rgb.y + 0.312f*rgb.z;
    const float c = cosf(ang), s = sinf(ang);
    const float In = c*I - s*Q;
    const float Qn = s*I + c*Q;
    float3 out;
    out.x = Y + 0.956f*In + 0.621f*Qn;
    out.y = Y - 0.272f*In - 0.647f*Qn;
    out.z = Y - 1.106f*In + 1.703f*Qn;
    return out;
}

// Post-color pass (edge glow and orbit tint)
__device__ __forceinline__ float3 applyEyeCandy(float3 col, float edgeW, float orbitW) {
    // Hue drift
    float3 c = hueRotateYIQ(col, d_phase);

    // Edge glow toward white
    if (d_glow > 1e-6f) {
        float g = d_glow * edgeW;
        c.x = fminf(1.0f, c.x + g * (1.0f - c.x));
        c.y = fminf(1.0f, c.y + g * (1.0f - c.y));
        c.z = fminf(1.0f, c.z + g * (1.0f - c.z));
    }

    // Orbit tint mix
    if (d_tintMix > 1e-6f && d_orbitK > 0.0f) {
        float m = d_tintMix * orbitW;
        c.x = (1.0f - m) * c.x + m * d_tint.x;
        c.y = (1.0f - m) * c.y + m * d_tint.y;
        c.z = (1.0f - m) * c.z + m * d_tint.z;
    }
    return c;
}

// ---------- Iteration (CHUNKed) ---------------------------------------------
// Pass 1: warmup with light periodicity + dz/dc + orbit trap
__device__ __forceinline__ int iterate_warmup_noLoop(
    float cr, float ci, int maxSteps,
    float& x, float& y, bool& interiorFlag,
    float& dx, float& dy, float& trapMinR2)
{
    x = 0.0f; y = 0.0f; dx = 0.0f; dy = 0.0f;
    trapMinR2 = CUDART_INF_F;
    int it = 0;
    interiorFlag = false;

    float px = x, py = y; // periodicity ref
    int   pc = 0, close_hits = 0;

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

            float x2 = x * x;
            float y2 = y * y;
            if (x2 + y2 > 4.0f) { active = false; ++pc; continue; }

            // orbit trap: min r2 to fixed point
            float tx = x - ORBIT_CX, ty = y - ORBIT_CY;
            float tr2 = tx*tx + ty*ty;
            trapMinR2 = fminf(trapMinR2, tr2);

            // dz/dc = 2*z*dz/dc + 1
            float ndx = 2.0f * (x * dx - y * dy) + 1.0f;
            float ndy = 2.0f * (x * dy + y * dx);

            float xt = fmaf(x, x, -y2) + cr;
            y = fmaf(2.0f * x, y, ci);
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

// Survivor payload
struct Survivor { float x, y, dx, dy, cr, ci, trapMinR2; int it, idx; };

// Pass-2 slice iteration
struct SliceResult { int it; float x, y, dx, dy, trapMinR2; bool escaped; bool interior; float de; };

__device__ __forceinline__ SliceResult iterate_finish_slice(
    float cr, float ci, int start_it, int maxIter,
    float x, float y, float dx, float dy, float trapMinR2,
    int sliceSteps)
{
    if (insideMainCardioidOrBulb(cr, ci)) {
        return { maxIter, x, y, dx, dy, trapMinR2, false, true, 0.0f };
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
                // Distance Estimator at escape
                float r   = sqrtf(x2 + y2);
                float dd  = fmaxf(1e-30f, sqrtf(dx*dx + dy*dy));
                deOut     = (r > 0.0f) ? (r * logf(r) / dd) : 0.0f;
                active    = false; escaped = true; ++pc; continue;
            }

            // orbit trap update
            float tx = x - ORBIT_CX, ty = y - ORBIT_CY;
            float tr2 = tx*tx + ty*ty;
            trapMinR2 = fminf(trapMinR2, tr2);

            // dz/dc
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
    return { it, x, y, dx, dy, trapMinR2, escaped, interior, deOut };
}

// ---------- Kernel: Pass 1 (warmup + compaction) ----------------------------
__global__ __launch_bounds__(256, 2)
void mandelbrotPass1Warmup(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    Survivor* __restrict__ surv, int* __restrict__ survCount,
    int w, int h, float zoom, float2 offset,
    int maxIter, float pixR)
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

    float zx=0.0f, zy=0.0f, dx=0.0f, dy=0.0f; bool interior = false; float trapMinR2 = CUDART_INF_F;
    int itWarm = iterate_warmup_noLoop(c.x, c.y, warmupSteps, zx, zy, interior, dx, dy, trapMinR2);

    if (interior) {
        out[idx]     = make_uchar4(0,0,0,255);
        iterOut[idx] = maxIter;
        return;
    }

    const float r2 = zx*zx + zy*zy;
    const bool escaped = (itWarm < warmupSteps) && (r2 > 4.0f);

    if (escaped) {
        // coverage via DE vs pixel radius
        const float r   = sqrtf(r2);
        const float dd  = fmaxf(1e-30f, sqrtf(dx*dx + dy*dy));
        const float de  = (r > 0.0f) ? (r * logf(r) / dd) : 0.0f;
        const float cov = smooth01(de / (AA_K * pixR));
        const float edgeW = 1.0f - cov;

        // orbit weight from trap min distance
        const float trapW = __expf(-d_orbitK * fmaxf(0.0f, sqrtf(trapMinR2)));

        float3 col = otter::shade(itWarm, maxIter, zx, zy, kPalette, kStripeF, kStripeAmp, kGamma);
        col = applyEyeCandy(col, edgeW, trapW);

        out[idx] = make_uchar4(
            (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
            (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
            255);
        iterOut[idx] = itWarm;
        return;
    }

    // survivors: black now to avoid ghosting
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

    Survivor s; s.x = zx; s.y = zy; s.dx = dx; s.dy = dy; s.cr = c.x; s.ci = c.y; s.trapMinR2 = trapMinR2; s.it = itWarm; s.idx = idx;
    surv[base + prefix] = s;
}

// ---------- Kernel: Pass 2 (slice + compaction) -----------------------------
__global__ __launch_bounds__(128, 2)
void mandelbrotPass2Slice(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    const Survivor* __restrict__ survIn, int survInCount,
    Survivor* __restrict__ survOut, int* __restrict__ survOutCount,
    int maxIter, int sliceIt, float pixR)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= survInCount) return;

    Survivor s = survIn[t];

    if (insideMainCardioidOrBulb(s.cr, s.ci)) {
        out[s.idx]     = make_uchar4(0,0,0,255);
        iterOut[s.idx] = maxIter;
        return;
    }

    SliceResult r = iterate_finish_slice(s.cr, s.ci, s.it, maxIter, s.x, s.y, s.dx, s.dy, s.trapMinR2, sliceIt);

    if (r.escaped) {
        const float cov   = smooth01(r.de / (AA_K * pixR));
        const float edgeW = 1.0f - cov;
        const float trapW = __expf(-d_orbitK * fmaxf(0.0f, sqrtf(r.trapMinR2)));

        float3 col = otter::shade(r.it, maxIter, r.x, r.y, kPalette, kStripeF, kStripeAmp, kGamma);
        col = applyEyeCandy(col, edgeW, trapW);

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

    // not finished yet: keep black and carry state
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

    Survivor ns; ns.x = r.x; ns.y = r.y; ns.dx = r.dx; ns.dy = r.dy; ns.cr = s.cr; ns.ci = s.ci; ns.trapMinR2 = r.trapMinR2; ns.it = r.it; ns.idx = s.idx;
    survOut[base + prefix] = ns;
}

// ---------- ENTROPY & CONTRAST ----------------------------------------------
__global__ void entropyKernel(
    const int* it, float* eOut, int w, int h, int tile, int maxIter)
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
    const float* e, float* cOut, int tilesX, int tilesY)
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

// ---------- Host wrappers ----------------------------------------------------
void computeCudaEntropyContrast(
    const int* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter)
{
    const bool doLog = (Settings::performanceLogging || Settings::debugLogging);

    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;

    cudaMemset(d_e, 0, tilesX * tilesY * sizeof(float));

    if (doLog) {
        cudaEvent_t evStart, evMid, evEnd;
        cudaEventCreate(&evStart); cudaEventCreate(&evMid); cudaEventCreate(&evEnd);

        cudaEventRecord(evStart, 0);
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        cudaEventRecord(evMid, 0);

        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
        cudaEventRecord(evEnd, 0);
        cudaEventSynchronize(evEnd);

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
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    }
}

// Persistent buffers for sliced finish
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

    dim3 block(32, 8);
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    ensureSurvivorCapacity(size_t(w) * size_t(h));

    const int warmupIt = chooseWarmupIt(maxIter);
    cudaMemcpyToSymbol(d_warmup_it, &warmupIt, sizeof(int), 0, cudaMemcpyHostToDevice);

    // Eye-candy phase & tint (time-based drift)
    static clk::time_point s_last = clk::now();
    static float s_phase = 0.0f;
    auto now = clk::now();
    double dt = std::chrono::duration<double>(now - s_last).count();
    s_last = now;

    // rotate ~0.35 rad/sec
    s_phase += float(dt) * 0.35f;
    if (s_phase > CUDART_PI_F) s_phase -= 2.0f * CUDART_PI_F;
    cudaMemcpyToSymbol(d_phase, &s_phase, sizeof(float));

    // subtle tint pulsing
    float3 tint = { 0.55f + 0.10f * std::sin(0.9f * s_phase),
                    0.25f + 0.08f * std::sin(1.5f * s_phase + 1.7f),
                    0.80f + 0.05f * std::sin(1.1f * s_phase + 0.4f) };
    cudaMemcpyToSymbol(d_tint, &tint, sizeof(float3));

    const float glow = 0.15f;  cudaMemcpyToSymbol(d_glow,   &glow,   sizeof(float));
    const float oK   = 6.0f;   cudaMemcpyToSymbol(d_orbitK, &oK,     sizeof(float));
    const float tMix = 0.20f;  cudaMemcpyToSymbol(d_tintMix,&tMix,   sizeof(float));

    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] warmup_it=%d prev_survivors=%.2f%% phase=%.3f", warmupIt, g_prevSurvivorsPct, s_phase);
    }

    cudaFuncSetCacheConfig(mandelbrotPass1Warmup, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mandelbrotPass2Slice,  cudaFuncCachePreferL1);

    // pixel footprint in c-plane (half diagonal)
    const float scale = 1.0f / zoom;
    const float spanX = 3.5f * scale;
    const float spanY = spanX * (float)h / (float)w;
    const float pixDiag = std::sqrt((spanX / w) * (spanX / w) + (spanY / h) * (spanY / h));
    const float pixR  = fmaxf(1e-25f, 0.5f * pixDiag);

    const double frameBudgetMs  = frameBudgetMsFromSettings();
    const double kernelBudgetMs = frameBudgetMs * KERNEL_BUDGET_FRACTION;
    const auto   hostStart      = clk::now();

    // Pass 1
    cudaMemset(g_dSurvCountA, 0, sizeof(int));
    mandelbrotPass1Warmup<<<grid, block>>>(out, d_it, g_dSurvivorsA, g_dSurvCountA, w, h, zoom, offset, maxIter, pixR);

    int h_survA = 0;
    cudaMemcpy(&h_survA, g_dSurvCountA, sizeof(int), cudaMemcpyDeviceToHost);
    const double survPct = (double)h_survA * 100.0 / (double(w) * double(h));
    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] survivors=%d (%.2f%% of %d)", h_survA, survPct, w*h);
    }
    g_prevSurvivorsPct = survPct;

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

    // Pass 2 (sliced)
    int threads = 128;
    int slice   = 0;
    int sliceIt = FINISH_SLICE_IT;

    Survivor* curBuf = g_dSurvivorsA;
    Survivor* nxtBuf = g_dSurvivorsB;
    int*      curCnt = g_dSurvCountA;
    int*      nxtCnt = g_dSurvCountB;
    int       h_cur  = h_survA;

    float emaDrop = 0.2f;

    while (h_cur > 0 && slice < MAX_SLICES) {
        double elapsedMs = std::chrono::duration<double, std::milli>(clk::now() - hostStart).count();
        if (elapsedMs >= kernelBudgetMs) {
            if (doLog) LUCHS_LOG_HOST("[PERF] budget_exhausted before slice %d: elapsed=%.3f ms budget=%.3f ms", slice, elapsedMs, kernelBudgetMs);
            break;
        }

        cudaMemset(nxtCnt, 0, sizeof(int));
        int blocks  = (h_cur + threads - 1) / threads;

        mandelbrotPass2Slice<<<blocks, threads>>>(
            out, d_it, curBuf, h_cur, nxtBuf, nxtCnt, maxIter, sliceIt, pixR);

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
            break;
        }

        const int drop = h_cur - h_next;
        const float dropPct = (h_cur > 0) ? float(drop) / float(h_cur) : 1.0f;
        emaDrop = (1.0f - DROP_EMA_ALPHA) * emaDrop + DROP_EMA_ALPHA * dropPct;

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

    if (doLog) {
        double totalMs = std::chrono::duration<double, std::milli>(clk::now() - hostStart).count();
        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] mandelbrot (hybrid-sliced): total=%.3f ms (budget=%.3f ms)", totalMs, kernelBudgetMs);
        } else if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[TIME] Mandelbrot Sliced | Total %.3f ms", totalMs);
        }
    }
}
