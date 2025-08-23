// =============================== core_kernel.cu ===============================
// 2-pass Mandelbrot (Warmup + Sliced Survivor Finish)
// Features:
// - Metric AA via distance estimator (no supersampling).
// - Consolidated post-fx: hue rotation, edge glow, orbit tint.
// - Circular wavy hotspot ripples (screen-space, soft, animated).
// - Angle-crest + orbit-trap gating; budget-aware sliced finish.
// All comments/strings ASCII-only.

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

namespace {
// ------------------------------- tuning --------------------------------------
constexpr int   WARP_CHUNK        = 64;
static_assert((WARP_CHUNK % 32) == 0, "WARP_CHUNK must be multiple of 32");

constexpr int   LOOP_CHECK_EVERY  = 16;
constexpr float LOOP_EPS2         = 1e-6f;
constexpr int   LOOP_REQ_HITS     = 1;

constexpr int   P1_LOOP_EVERY     = 48;
constexpr float P1_LOOP_EPS2      = 2e-7f;
constexpr int   P1_LOOP_REQ_HITS  = 1;

constexpr int   WARMUP_IT_BASE    = 1024;
constexpr int   FINISH_SLICE_IT   = 1024;
constexpr int   MAX_SLICES        = 64;

constexpr otter::Palette kPalette = otter::Palette::Glacier;
constexpr float kStripeF   = 3.0f;
constexpr float kStripeAmp = 0.10f;

#if defined(OTTER_FRAMEBUFFER_SRGB) && (OTTER_FRAMEBUFFER_SRGB)
  constexpr float kGamma = 1.0f;
#else
  constexpr float kGamma = 2.2f;
#endif

constexpr double FRAME_BUDGET_FALLBACK_MS = 16.6667;
constexpr double KERNEL_BUDGET_FRAC       = 0.62;
constexpr double MIN_BUDGET_MS            = 6.0;

constexpr float DROP_EMA_ALPHA      = 0.25f;
constexpr float DROP_UPPER_BACKOFF  = 0.30f;
constexpr float DROP_LOWER_ACCEL    = 0.005f;

constexpr float DEEP_EPS2           = 1e-10f;
constexpr int   DEEP_REQ_HITS       = 3;

constexpr float PI2 = 6.28318530717958647692f;

// --------------------------- device constants --------------------------------
struct EffectsParams {
    // AA scale (coverage from DE)
    float aaK;

    // hue rotation
    float huePhase;            // radians

    // edge glow
    float glowAmount;          // 0..1

    // orbit tint
    float orbitK;              // exp(-k * sqrt(minR2))
    float tintMix;             // 0..1
    float3 tint;               // RGB 0..1

    // angle-crest detector
    float crestF;              // frequency for cos(F * angle)
    float crestSharp;          // exponent to sharpen peaks

    // green hotspot pulses (intensity base/gating)
    float hotspotStrength;     // intensity
    float hotspotRate;         // pulses per unit phase
    float hotspotTau;          // decay in phase units
    float3 hotColor;           // RGB 0..1

    // screen-space sprite layout
    float dotCellPx;           // grid cell size in pixels
    float dotRadiusPx;         // reference radius in pixels

    // wave-ring controls (radial ripples)
    float waveLambdaPx;        // wavelength in pixels
    float waveWidthPx;         // gaussian envelope sigma (px)
    float waveSpeed;           // phase speed multiplier
    float waveGain;            // extra gain for the ring crest
};
__device__ __constant__ EffectsParams d_fx;

// adaptive warmup iterations
__device__ __constant__ int d_warmup_it = WARMUP_IT_BASE;

// ------------------------------- helpers -------------------------------------
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

__device__ __forceinline__ float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}
__device__ __forceinline__ float smooth01(float x) {
    x = clamp01(x);
    return x * x * (3.0f - 2.0f * x);
}
__device__ __forceinline__ float fractf(float x) {
    return x - floorf(x);
}
__device__ __forceinline__ float hash2(float a, float b) {
    float n = a * 12.9898f + b * 78.233f;
    return fractf(sinf(n) * 43758.5453f);
}

// RGB<->YIQ helpers for hue rotation
__device__ __forceinline__ float3 rgb2yiq(float3 c) {
    float Y = 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
    float I = 0.596f * c.x - 0.274f * c.y - 0.322f * c.z;
    float Q = 0.211f * c.x - 0.523f * c.y + 0.312f * c.z;
    return make_float3(Y, I, Q);
}
__device__ __forceinline__ float3 yiq2rgb(float3 c) {
    float r = c.x + 0.956f * c.y + 0.621f * c.z;
    float g = c.x - 0.272f * c.y - 0.647f * c.z;
    float b = c.x - 1.106f * c.y + 1.703f * c.z;
    return make_float3(fminf(fmaxf(r, 0.0f), 1.0f),
                       fminf(fmaxf(g, 0.0f), 1.0f),
                       fminf(fmaxf(b, 0.0f), 1.0f));
}
__device__ __forceinline__ float3 hueRotateYIQ(float3 rgb, float phaseRad) {
    if (fabsf(phaseRad) < 1e-6f) return rgb;
    float3 yiq = rgb2yiq(rgb);
    float c = cosf(phaseRad), s = sinf(phaseRad);
    float I = yiq.y * c - yiq.z * s;
    float Q = yiq.y * s + yiq.z * c;
    return yiq2rgb(make_float3(yiq.x, I, Q));
}

// coverage from distance estimator
__device__ __forceinline__ float coverage_from_de(float r2, float dx, float dy, float pixR) {
    float r  = sqrtf(fmaxf(r2, 0.0f));
    float dd = sqrtf(dx*dx + dy*dy);
    dd = (dd > 1e-30f) ? dd : 1e-30f;
    float de = (r > 0.0f) ? (r * logf(r) / dd) : 0.0f;
    float cov = smooth01(de / (d_fx.aaK * fmaxf(pixR, 1e-30f)));
    return cov;
}

// angle-crest weight from escape z = (zx, zy)
__device__ __forceinline__ float crest_weight(float zx, float zy) {
    float ang = atan2f(zy, zx);              // -pi..pi
    float v   = 0.5f * (cosf(d_fx.crestF * ang) + 1.0f); // 0..1
    return powf(clamp01(v), fmaxf(1.0f, d_fx.crestSharp));
}

// wavy circular sprite (radial ripple), deterministic per cell; returns 0..1
__device__ __forceinline__ float dot_mask_wave_screen(
    int xPix, int yPix, int w, int h,
    float cellPx, float radiusPx,
    float lambdaPx, float sigmaPx, float speed, float phase)
{
    // cell indices
    float gx = float(xPix) / fmaxf(cellPx, 1.0f);
    float gy = float(yPix) / fmaxf(cellPx, 1.0f);
    int ix = int(floorf(gx));
    int iy = int(floorf(gy));

    // jittered center inside cell (stable)
    float jx = hash2(float(ix), float(iy)) - 0.5f; // [-0.5, 0.5)
    float jy = hash2(float(iy), float(ix)) - 0.5f;
    float cx = (float(ix) + 0.5f + 0.35f * jx) * cellPx;
    float cy = (float(iy) + 0.5f + 0.35f * jy) * cellPx;

    // pixel center
    float px = float(xPix) + 0.5f;
    float py = float(yPix) + 0.5f;

    float dx = px - cx;
    float dy = py - cy;
    float d  = sqrtf(dx*dx + dy*dy);

    // gaussian envelope around radiusPx (soft window)
    float sig = fmaxf(0.5f, sigmaPx);
    float g   = __expf(-((d - radiusPx)*(d - radiusPx)) / (2.0f * sig * sig));

    // radial sine wave centered at radiusPx
    float k   = PI2 / fmaxf(1.0f, lambdaPx);
    // slight per-cell phase offset for dephasing
    float phiJ = PI2 * hash2(float(ix)*3.1f, float(iy)*7.7f);
    float arg  = k * (d - radiusPx) - speed * phase + phiJ;

    float wave = 0.5f * (cosf(arg) + 1.0f); // 0..1
    // sharpen crest
    wave = powf(wave, 3.0f);

    return clamp01(g * wave);
}

// consolidated post-fx with wavy dots
__device__ __forceinline__ float3 apply_post_fx(
    float3 base, float coverage, float trapMinR2,
    float cr, float ci, float zx, float zy, float phase01,
    int xPix, int yPix, int w, int h)
{
    // edge weight: 0 in interior, 1 near thin edge
    float edgeW = 1.0f - clamp01(coverage);

    // hue rotation
    base = hueRotateYIQ(base, d_fx.huePhase);

    // edge glow toward white
    if (d_fx.glowAmount > 1e-6f) {
        float g = d_fx.glowAmount * edgeW;
        base.x = fminf(1.0f, base.x + g * (1.0f - base.x));
        base.y = fminf(1.0f, base.y + g * (1.0f - base.y));
        base.z = fminf(1.0f, base.z + g * (1.0f - base.z));
    }

    // orbit tint
    float trapW = __expf(-d_fx.orbitK * fmaxf(0.0f, sqrtf(fmaxf(trapMinR2, 0.0f))));
    if (d_fx.tintMix > 1e-6f) {
        float m = d_fx.tintMix * trapW;
        base.x = (1.0f - m) * base.x + m * d_fx.tint.x;
        base.y = (1.0f - m) * base.y + m * d_fx.tint.y;
        base.z = (1.0f - m) * base.z + m * d_fx.tint.z;
    }

    // hotspot gate: either orbit proximity OR angle crest
    float crestW = crest_weight(zx, zy);
    float gate   = fmaxf(trapW, crestW);

    // pulsation envelope (phase01 in 0..1)
    float s   = fractf(phase01 * d_fx.hotspotRate + hash2(cr, ci));
    float env = __expf(-s / fmaxf(1e-3f, d_fx.hotspotTau));

    // wavy circular sprite in screen space
    float dotM = dot_mask_wave_screen(
        xPix, yPix, w, h,
        d_fx.dotCellPx, d_fx.dotRadiusPx,
        d_fx.waveLambdaPx, d_fx.waveWidthPx,
        d_fx.waveSpeed, phase01 * PI2);

    // final hotspot (additive with gain, modulated by gates)
    if (d_fx.hotspotStrength > 1e-6f) {
        float hot = d_fx.hotspotStrength * d_fx.waveGain * gate * edgeW * env * dotM;
        base.x = fminf(1.0f, base.x + hot * d_fx.hotColor.x);
        base.y = fminf(1.0f, base.y + hot * d_fx.hotColor.y);
        base.z = fminf(1.0f, base.z + d_fx.hotColor.z * hot);
    }

    return base;
}

// --------------------------- iteration (chunked) -----------------------------
// Pass 1: warmup with light periodicity + dz/dc + orbit-trap
__device__ __forceinline__ int iterate_warmup_noLoop(
    float cr, float ci, int maxSteps,
    float& x, float& y, bool& interiorFlag,
    float& dx, float& dy, float& trapMinR2)
{
    x = 0.0f; y = 0.0f; dx = 0.0f; dy = 0.0f;
    trapMinR2 = CUDART_INF_F;
    int it = 0;
    interiorFlag = false;

    const float trX = -0.745f, trY = 0.186f;

    float px = x, py = y;
    int pc = 0, close_hits = 0;

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

            float x2 = x * x, y2 = y * y;
            if (x2 + y2 > 4.0f) { active = false; ++pc; continue; }

            // orbit trap min r^2
            float tx = x - trX, ty = y - trY;
            float r2t = tx*tx + ty*ty;
            trapMinR2 = fminf(trapMinR2, r2t);

            // dz/dc update: d <- 2*z*d + 1
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
                        active = false;
                        interiorFlag = true;
                        it = maxSteps;
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

// survivor payload
struct Survivor { float x, y, dx, dy, cr, ci, trapMinR2; int it, idx; };

// pass-2 slice iteration
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

    const float trX = -0.745f, trY = 0.186f;

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
                float r  = sqrtf(x2 + y2);
                float dd = sqrtf(dx*dx + dy*dy) + 1e-30f;
                deOut    = (r > 0.0f) ? (r * logf(r) / dd) : 0.0f;
                active   = false; escaped = true; ++pc; continue;
            }

            // orbit trap update
            float tx = x - trX, ty = y - trY;
            float r2t = tx*tx + ty*ty;
            trapMinR2 = fminf(trapMinR2, r2t);

            // dz/dc update
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

// ----------------------- pass 1 kernel (warmup/compact) ----------------------
__global__ __launch_bounds__(256, 2)
void mandelbrotPass1Warmup(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    Survivor* __restrict__ surv, int* __restrict__ survCount,
    int w, int h, float zoom, float2 offset,
    int maxIter, float pixR)
{
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
        return;
    }

    const int warmupSteps = d_warmup_it;

    float zx=0.0f, zy=0.0f, dx=0.0f, dy=0.0f, trapMinR2=CUDART_INF_F;
    bool interior = false;
    int itWarm = iterate_warmup_noLoop(c.x, c.y, warmupSteps, zx, zy, interior, dx, dy, trapMinR2);

    if (interior) {
        out[idx]     = make_uchar4(0,0,0,255);
        iterOut[idx] = maxIter;
        return;
    }

    const float r2 = zx*zx + zy*zy;
    const bool escaped = (itWarm < warmupSteps) && (r2 > 4.0f);

    float phase01 = fractf(d_fx.huePhase * (0.5f / CUDART_PI_F) + 0.5f);

    if (escaped) {
        float cov = coverage_from_de(r2, dx, dy, pixR);
        float3 col = otter::shade(itWarm, maxIter, zx, zy, kPalette, kStripeF, kStripeAmp, kGamma);
        col = apply_post_fx(col, cov, trapMinR2, c.x, c.y, zx, zy, phase01, xPix, yPix, w, h);

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

    Survivor s; s.x = zx; s.y = zy; s.dx = dx; s.dy = dy;
    s.cr = c.x; s.ci = c.y; s.it = itWarm; s.idx = idx; s.trapMinR2 = trapMinR2;
    surv[base + prefix] = s;
}

// ------------------------- pass 2 kernel (sliced) ----------------------------
__global__ __launch_bounds__(128, 2)
void mandelbrotPass2Slice(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    const Survivor* __restrict__ survIn, int survInCount,
    Survivor* __restrict__ survOut, int* __restrict__ survOutCount,
    int maxIter, int sliceIt, float pixR,
    int w, int h) // pass w,h to reconstruct pixel coords
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= survInCount) return;

    Survivor s = survIn[t];

    if (insideMainCardioidOrBulb(s.cr, s.ci)) {
        out[s.idx]     = make_uchar4(0,0,0,255);
        iterOut[s.idx] = maxIter;
        return;
    }

    SliceResult r = iterate_finish_slice(
        s.cr, s.ci, s.it, maxIter, s.x, s.y, s.dx, s.dy, s.trapMinR2, sliceIt);

    float phase01 = fractf(d_fx.huePhase * (0.5f / CUDART_PI_F) + 0.5f);

    // reconstruct pixel coords from linear index
    int xPix = s.idx % w;
    int yPix = s.idx / w;

    if (r.escaped) {
        float cov = coverage_from_de(r.x*r.x + r.y*r.y, r.dx, r.dy, pixR);
        float3 col = otter::shade(r.it, maxIter, r.x, r.y, kPalette, kStripeF, kStripeAmp, kGamma);
        col = apply_post_fx(col, cov, r.trapMinR2, s.cr, s.ci, r.x, r.y, phase01, xPix, yPix, w, h);

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

    // survivor continues: write black and compact
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

    Survivor ns; ns.x = r.x; ns.y = r.y; ns.dx = r.dx; ns.dy = r.dy;
    ns.trapMinR2 = r.trapMinR2; ns.cr = s.cr; ns.ci = s.ci; ns.it = r.it; ns.idx = s.idx;
    survOut[base + prefix] = ns;
}

// ----------------- entropy & contrast (coarse metrics) -----------------------
__global__ void entropyKernel(
    const int* it, float* eOut,
    int w, int h, int tile, int maxIter)
{
    int tX = blockIdx.x, tY = blockIdx.y;
    int startX = tX * tile, startY = tY * tile;

    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;
    int tileIndex = tY * tilesX + tX;

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
    }
}

__global__ void contrastKernel(
    const float* e, float* cOut,
    int tilesX, int tilesY)
{
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

    cOut[idx] = (cnt > 0) ? (sum / cnt) : 0.0f;
}

// ---------------- host wrappers: metrics -------------------------------------
} // namespace

void computeCudaEntropyContrast(
    const int* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter)
{
    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;

    cudaMemset(d_e, 0, tilesX * tilesY * sizeof(float));

    if (Settings::performanceLogging || Settings::debugLogging) {
        cudaEvent_t evStart, evMid, evEnd;
        cudaEventCreate(&evStart); cudaEventCreate(&evMid); cudaEventCreate(&evEnd);

        cudaEventRecord(evStart, 0);
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        cudaEventRecord(evMid, 0);

        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
        cudaEventRecord(evEnd, 0);
        cudaEventSynchronize(evEnd);

        float ms1=0.f, ms2=0.f;
        cudaEventElapsedTime(&ms1, evStart, evMid);
        cudaEventElapsedTime(&ms2, evMid, evEnd);

        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] entropy=%.3f ms contrast=%.3f ms", ms1, ms2);
        } else {
            LUCHS_LOG_HOST("[TIME] Entropy %.3f ms | Contrast %.3f ms", ms1, ms2);
        }

        cudaEventDestroy(evStart); cudaEventDestroy(evMid); cudaEventDestroy(evEnd);
    } else {
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    }
}

// ---------------- host wrapper: 2-pass sliced renderer -----------------------
namespace {
    using clk = std::chrono::high_resolution_clock;

    struct Survivor; // fwd (already defined above in TU)
    struct DevicePools {
        Survivor* A = nullptr;
        Survivor* B = nullptr;
        int*      cntA = nullptr;
        int*      cntB = nullptr;
        size_t    cap = 0;
    };
    DevicePools g_pools;
    double      g_prevSurvivorsPct = -1.0;

    void ensureSurvivorCapacity(size_t need) {
        if (need <= g_pools.cap) return;
        if (g_pools.A)    cudaFree(g_pools.A);
        if (g_pools.B)    cudaFree(g_pools.B);
        if (g_pools.cntA) cudaFree(g_pools.cntA);
        if (g_pools.cntB) cudaFree(g_pools.cntB);
        cudaMalloc(&g_pools.A,    need * sizeof(Survivor));
        cudaMalloc(&g_pools.B,    need * sizeof(Survivor));
        cudaMalloc(&g_pools.cntA, sizeof(int));
        cudaMalloc(&g_pools.cntB, sizeof(int));
        g_pools.cap = need;
    }

    int chooseWarmupIt(int maxIter) {
        int warm = WARMUP_IT_BASE;
        if (g_prevSurvivorsPct >= 90.0)      warm = std::min(maxIter, WARMUP_IT_BASE * 3);
        else if (g_prevSurvivorsPct >= 80.0) warm = std::min(maxIter, WARMUP_IT_BASE * 2);
        else if (g_prevSurvivorsPct >= 60.0) warm = std::min(maxIter, (WARMUP_IT_BASE * 3) / 2);
        warm = std::min(warm, std::max(64, maxIter / 3));
        return warm;
    }

    double frameBudgetMsFromSettings() {
        if (Settings::capFramerate && Settings::capTargetFps > 0) {
            return std::max(MIN_BUDGET_MS, 1000.0 / double(Settings::capTargetFps));
        }
        return std::max(MIN_BUDGET_MS, FRAME_BUDGET_FALLBACK_MS);
    }
} // namespace

void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using namespace std::chrono;

    // default FX params (can be adjusted live before memcpy)
    EffectsParams fx{};
    fx.aaK             = 2.0f;
    fx.huePhase        = float(duration<double>(clk::now().time_since_epoch()).count() * 0.6f); // slow drift
    fx.glowAmount      = 0.22f;  // stronger subtle glow
    fx.orbitK          = 6.0f;
    fx.tintMix         = 0.25f;
    fx.tint            = make_float3(0.62f, 0.40f, 0.95f);   // soft violet
    fx.crestF          = 9.0f;   // number of bright crests around lobes
    fx.crestSharp      = 4.0f;   // sharpness of crests
    fx.hotspotStrength = 0.85f;  // boosted hotspot intensity
    fx.hotspotRate     = 1.2f;
    fx.hotspotTau      = 0.35f;
    fx.hotColor        = make_float3(0.05f, 1.0f, 0.10f);    // green
    fx.dotCellPx       = 10.0f;  // denser grid
    fx.dotRadiusPx     = 5.5f;   // slightly larger
    fx.waveLambdaPx    = 5.0f;   // ring spacing
    fx.waveWidthPx     = 8.0f;   // gaussian sigma around radius
    fx.waveSpeed       = 1.25f;  // ripple motion speed
    fx.waveGain        = 1.8f;   // stronger crest
    cudaMemcpyToSymbol(d_fx, &fx, sizeof(EffectsParams), 0, cudaMemcpyHostToDevice);

    // pixel footprint radius in c-plane (half pixel diagonal)
    const float scale = 1.0f / zoom;
    const float spanX = 3.5f * scale;
    const float spanY = spanX * (float)h / (float)w;
    const float pixR  = 0.5f * sqrtf((spanX / w)*(spanX / w) + (spanY / h)*(spanY / h));

    // kernels grid/block
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    ensureSurvivorCapacity(size_t(w) * size_t(h));

    // adaptive warmup
    const int warmupIt = chooseWarmupIt(maxIter);
    cudaMemcpyToSymbol(d_warmup_it, &warmupIt, sizeof(int), 0, cudaMemcpyHostToDevice);

    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] warmup_it=%d prev_survivors=%.2f%%", warmupIt, g_prevSurvivorsPct);
    }

    cudaFuncSetCacheConfig(mandelbrotPass1Warmup, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mandelbrotPass2Slice,  cudaFuncCachePreferL1);

    // frame budget pacing (host-side only)
    const double frameBudgetMs  = frameBudgetMsFromSettings();
    const double kernelBudgetMs = frameBudgetMs * KERNEL_BUDGET_FRAC;
    const auto   hostStart      = clk::now();

    // pass 1
    cudaMemset(g_pools.cntA, 0, sizeof(int));
    mandelbrotPass1Warmup<<<grid, block>>>(out, d_it, g_pools.A, g_pools.cntA, w, h, zoom, offset, maxIter, pixR);

    int h_survA = 0;
    cudaMemcpy(&h_survA, g_pools.cntA, sizeof(int), cudaMemcpyDeviceToHost); // waits for P1
    const double survPct = (double)h_survA * 100.0 / (double(w) * double(h));
    if (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] survivors=%d (%.2f%% of %d)", h_survA, survPct, w*h);
    }
    g_prevSurvivorsPct = survPct;

    double p1Ms = duration<double, std::milli>(clk::now() - hostStart).count();
    if (h_survA <= 0) {
        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] mandelbrot (hybrid-sliced): total=%.3f ms", p1Ms);
        } else if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[TIME] Mandelbrot Sliced | Total %.3f ms", p1Ms);
        }
        return;
    }
    if (p1Ms > kernelBudgetMs && Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] budget_hit after P1: p1=%.3f ms budget=%.3f ms -> defer P2", p1Ms, kernelBudgetMs);
    }

    // pass 2 (sliced, budget-aware)
    int threads = 128;
    int slice   = 0;
    int sliceIt = FINISH_SLICE_IT;

    Survivor* curBuf = g_pools.A;
    Survivor* nxtBuf = g_pools.B;
    int*      curCnt = g_pools.cntA;
    int*      nxtCnt = g_pools.cntB;
    int       h_cur  = h_survA;

    float emaDrop = 0.2f;

    while (h_cur > 0 && slice < MAX_SLICES) {
        double elapsedMs = duration<double, std::milli>(clk::now() - hostStart).count();
        if (elapsedMs >= kernelBudgetMs) {
            if (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[PERF] budget_exhausted before slice %d: elapsed=%.3f ms budget=%.3f ms",
                               slice, elapsedMs, kernelBudgetMs);
            }
            break;
        }

        cudaMemset(nxtCnt, 0, sizeof(int));
        int blocks = (h_cur + threads - 1) / threads;

        mandelbrotPass2Slice<<<blocks, threads>>>(
            out, d_it, curBuf, h_cur, nxtBuf, nxtCnt, maxIter, sliceIt, pixR, w, h);

        int h_next = 0;
        cudaMemcpy(&h_next, nxtCnt, sizeof(int), cudaMemcpyDeviceToHost);

        elapsedMs = duration<double, std::milli>(clk::now() - hostStart).count();
        if (Settings::performanceLogging || Settings::debugLogging) {
            LUCHS_LOG_HOST("[PERF] slice=%d steps=%d survivors_in=%d survivors_out=%d elapsed=%.3f ms (budget=%.3f)",
                           slice, sliceIt, h_cur, h_next, elapsedMs, kernelBudgetMs);
        }
        if (elapsedMs >= kernelBudgetMs) {
            if (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[PERF] budget_stop at slice %d", slice);
            }
            std::swap(curBuf, nxtBuf); std::swap(curCnt, nxtCnt); h_cur = h_next; ++slice;
            break;
        }

        const int drop = h_cur - h_next;
        const float dropPct = (h_cur > 0) ? float(drop) / float(h_cur) : 1.0f;
        emaDrop = (1.0f - DROP_EMA_ALPHA) * emaDrop + DROP_EMA_ALPHA * dropPct;

        // adapt slice length
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

    if (Settings::performanceLogging || Settings::debugLogging) {
        double totalMs = duration<double, std::milli>(clk::now() - hostStart).count();
        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] mandelbrot (hybrid-sliced): total=%.3f ms (budget=%.3f ms)",
                           totalMs, kernelBudgetMs);
        } else {
            LUCHS_LOG_HOST("[TIME] Mandelbrot Sliced | Total %.3f ms", totalMs);
        }
    }
}
