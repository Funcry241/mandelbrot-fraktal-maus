// MAUS: core kernel with tiny device formatter (no CRT redeclare; bounded; ASCII)
// 🐭 Maus: „Schwarze Schnauze“ erweitert – analytisch (Cardioid/p=2) + selektiver Attraktor-Check p=3..8.
// 🦦 Otter: Läuft nur bei Innen-Kandidaten (billiger Vorfilter) → spart Iterationen ohne Bildänderung. (Bezug zu Otter)
// 🦊 Schneefuchs: Konservativ & numerisch stabil; zusätzlich seltener Periodizitätsprobe im Hauptloop. (Bezug zu Schneefuchs)
// 🐑 Schneefuchs: Warp-synchrones Escape & FMA – weniger Divergenz, weniger Instruktionen. (Bezug zu Schneefuchs)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include <chrono>
#include "common.hpp"
#include "luchs_device_format.hpp"   // tiny formatter (no snprintf)
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"

// --- Extended Nose Configuration (compile-time, conservative) ----------------
// (Kein settings.hpp Touch – bildneutral, nur Workload-Reduktion)
namespace {
constexpr int   NOSE_PREFILTER_STEPS   = 8;       // mini warmup to detect "likely interior"
constexpr float NOSE_PREFILTER_THRESH2 = 0.36f;   // r^2 < 0.36  (r < 0.6) nach Vorfilter
constexpr int   NOSE_P_MAX             = 8;       // check periods 3..8 (p=1,2 handled analytically)
constexpr int   NOSE_CYCLE_STEPS       = 32;      // etwas billiger als 48, bleibt konservativ
constexpr float NOSE_ESC2              = 4.0f;    // escape radius^2
constexpr float NOSE_EPS_CYCLE2        = 1e-10f;  // closeness threshold für z ~ z_{-p}
constexpr float NOSE_MU_THR            = 0.95f;   // |∏(2 z_k)| < thr ⇒ stabil
constexpr int   LOOP_CHECK_EVERY       = 16;      // Periodizitätsprobe-Frequenz im Hauptloop
constexpr float LOOP_EPS2              = 1e-8f;   // Nähe-Schwelle für Probe (etwas großzügiger)
} // namespace

// --- Helpers ----------------------------------------------------------------

__device__ __forceinline__ float fract(float x) {
    return x - floorf(x);
}

__device__ __forceinline__ uchar4 elegantColor(float t) {
    t = sqrtf(fminf(fmaxf(t, 0.0f), 1.0f));
    float rf = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.0f));
    float gf = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.33f));
    float bf = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.66f));
    unsigned char r = static_cast<unsigned char>(rf * 255.0f);
    unsigned char g = static_cast<unsigned char>(gf * 255.0f);
    unsigned char b = static_cast<unsigned char>(bf * 255.0f);
    return make_uchar4(r, g, b, 255);
}

__device__ __forceinline__ float2 pixelToComplex(
    float px, float py, int w, int h,
    float spanX, float spanY, float2 offset)
{
    return make_float2(
        (px / w - 0.5f) * spanX + offset.x,
        (py / h - 0.5f) * spanY + offset.y
    );
}

// --- Mandelbrot (baseline) ---------------------------------------------------

__device__ __forceinline__ int mandelbrotIterations_scalar(
    float x0, float y0, int maxIter,
    float& fx, float& fy)
{
    float x = 0.0f, y = 0.0f;
    int i = 0;
#pragma unroll 1
    while (x * x + y * y <= 4.0f && i < maxIter) {
        float xx = x * x;
        float yy = y * y;
        float xy = x * y;
        float xt = xx - yy + x0;
        y = 2.0f * xy + y0;
        x = xt;
        ++i;
    }
    fx = x;
    fy = y;
    return i;
}

// 🐑 Schneefuchs: Warp-synchronisierte Iteration mit Ballot – reduziert Divergenz.
// + seltene Periodizitätsprobe (alle LOOP_CHECK_EVERY steps) – bildneutraler Early-Out für sichere Innenbahnen.
__device__ __forceinline__ int mandelbrotIterations_warp(
    float cr, float ci, int maxIter, float& xr, float& xi)
{
    float x = 0.0f, y = 0.0f;
    int it = 0;

    // rare periodicity probe state
    float px = 0.0f, py = 0.0f; int pc = 0;
    int close_hits = 0; // zwei Treffer hintereinander → sicher innen

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif

    bool active = true;

#pragma unroll 1
    for (int k = 0; k < maxIter; ++k) {
        // Check escape radius BEFORE heavy math for escaped threads.
        float x2 = x * x;
        float y2 = y * y;
        if (active && (x2 + y2 <= 4.0f)) {
            // z = z^2 + c  with FMA to reduce ops and improve precision.
            float xt = fmaf(x, x, -y2) + cr;                  // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);                        // 2*x*y + ci
            x = xt;
            ++it;

            // ---- seltene Periodizitätsprobe (robuster, bildneutral) ----
            ++pc;
            if (pc == LOOP_CHECK_EVERY) {
                float dx = x - px, dy = y - py;
                if (dx*dx + dy*dy < LOOP_EPS2) {
                    if (++close_hits >= 2) { // zwei unabhängige "nahe Wiederholung"
                        active = false; it = maxIter;     // als Innenpunkt werten
                    }
                } else {
                    close_hits = 0; // Sequenz zurücksetzen
                }
                px = x; py = y; pc = 0;
            }
            // ----------------------------------------------------------------
        } else {
            active = false;
        }

        // Warp votes: break when all threads are inactive (escaped or finished).
        unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }

    xr = x; xi = y;
    return it;
}

// --- Farbe / Mapping ---------------------------------------------------------

__device__ __forceinline__ float3 hsvToRgb(float h, float s, float v) {
    float r, g, b;
    int i = int(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        default: r = v, g = p, b = q; break;
    }
    return make_float3(r, g, b);
}

__device__ float pseudoRandomWarze(float x, float y) {
    float r = sqrtf(x * x + y * y);
    float angle = atan2f(y, x);
    return 0.5f + 0.5f * __sinf(r * 6.0f + angle * 4.0f);
}

// Continuous Escape-Time (CEC) + Stripe
__device__ __forceinline__
void computeCEC(float zx, float zy, int it, int maxIt, float& nu, float& stripe)
{
    float norm = zx * zx + zy * zy;
    if (it >= maxIt) {
        nu = 1.0f;
        stripe = 0.0f;
        return;
    }
    // Schneefuchs: __log2f ist schnelle Approx.; fmaxf schützt Bereich.
    float mu = (float)it + 1.0f - __log2f(__log2f(fmaxf(norm, 1.000001f)));
    nu = fminf(fmaxf(mu / (float)maxIt, 0.0f), 1.0f);
    float fracv = fract(mu);
    stripe = powf(0.5f + 0.5f * __sinf(6.2831853f * fracv), 0.75f);
}

__device__ __forceinline__
float3 colorFractalDetailed(float2 c, float zx, float zy, int it, int maxIt)
{
    if (it >= maxIt) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float nu, stripe;
    computeCEC(zx, zy, it, maxIt, nu, stripe);

    float angle = atan2f(c.y, c.x);
    // 0.15915494f = 1/(2*pi)
    float hue   = fract(nu * 0.25f + angle * 0.08f * 0.15915494f);
    float val   = 0.3f + 0.7f * stripe;
    float sat   = 0.9f;
    return hsvToRgb(hue, sat, val);
}

// --- „Schwarze Schnauze“: Innenraum-Shortcuts -------------------------------
// Otter: Early-Out für Punkte sicher in der Menge – spart komplette Iteration. (Bezug zu Otter)
// Schneefuchs: Zwei exakte Tests (Hauptcardioide, period-2 Bulb). (Bezug zu Schneefuchs)
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y) {
    // Hauptcardioide
    float xm = x - 0.25f;
    float q  = xm * xm + y * y;
    if (q * (q + xm) <= 0.25f * y * y) return true;

    // period-2 Bulb um (-1,0) mit r=0.25
    float xp = x + 1.0f;
    if (xp * xp + y * y <= 0.0625f) return true;

    return false;
}

// 🐭 Maus: Cheap prefilter – aktiviere erweiterten Check nur bei Innen-Kandidat. (Bezug zu Otter/Schneefuchs)
__device__ __forceinline__
bool likelyInteriorAfterFewSteps(float cr, float ci, float& x, float& y)
{
    x = 0.0f; y = 0.0f;
#pragma unroll 1
    for (int i = 0; i < NOSE_PREFILTER_STEPS; ++i) {
        float x2 = x * x, y2 = y * y;
        if (x2 + y2 > NOSE_ESC2) return false; // sicher außen → teuren Check sparen
        float xt = fmaf(x, x, -y2) + cr;
        y = fmaf(2.0f * x, y, ci);
        x = xt;
    }
    // konservative Schwelle: "klar klein" nach wenigen Schritten
    return (x * x + y * y) < NOSE_PREFILTER_THRESH2;
}

// 🐑 Schneefuchs: Konservativer Attraktor-Check p=3..8, Start im vorgefilterten Zustand. (Bezug zu Schneefuchs)
__device__ __forceinline__
bool insideHigherPeriodFromState(float x, float y, float cr, float ci)
{
    float2 ring[NOSE_P_MAX];
    int rp = 0;

#pragma unroll 1
    for (int t = 0; t < NOSE_CYCLE_STEPS; ++t) {
        float x2 = x * x, y2 = y * y;
        if (x2 + y2 > NOSE_ESC2) return false; // außen

        ring[rp] = make_float2(x, y);

        // einen Schritt vorwärts
        float xt = fmaf(x, x, -y2) + cr;
        y = fmaf(2.0f * x, y, ci);
        x = xt;

        rp = (rp + 1) % NOSE_P_MAX;

        if (t >= NOSE_P_MAX) {
            // Perioden p=3..P_MAX prüfen (p=1,2 analytisch abgedeckt)
            for (int p = 3; p <= NOSE_P_MAX; ++p) {
                int idx_prev = rp - p;
                if (idx_prev < 0) idx_prev += NOSE_P_MAX; // reicht, da p<=NOSE_P_MAX
                float dx = x - ring[idx_prev].x;
                float dy = y - ring[idx_prev].y;
                if (dx * dx + dy * dy < NOSE_EPS_CYCLE2) {
                    // Multiplier μ = ∏(2 z_k) über p Zustände
                    float mx = 1.0f, my = 0.0f;
                    int idx = idx_prev;
#pragma unroll 1
                    for (int j = 0; j < p; ++j) {
                        float ax = mx, ay = my;
                        float bx = 2.0f * ring[idx].x, by = 2.0f * ring[idx].y;
                        mx = fmaf(ax, bx, -ay * by);
                        my = fmaf(ax, by,  ay * bx);
                        idx = (idx + 1) % NOSE_P_MAX;
                    }
                    float mu2 = mx * mx + my * my;
                    if (mu2 < NOSE_MU_THR * NOSE_MU_THR) {
                        return true; // stabiler Attraktor sicher → Innenpunkt
                    }
                }
            }
        }
    }
    return false; // unentschieden → normal iterieren
}

// --- Kernel ------------------------------------------------------------------

// 🐑 Schneefuchs: __restrict__-Aliase helfen dem Compiler ohne API-Änderung. (Bezug zu Schneefuchs)
__global__ void mandelbrotKernel(
    uchar4* out, int* iterOut,
    int w, int h, float zoom, float2 offset, int maxIter)
{
    const bool doLog = Settings::debugLogging;

    uchar4* __restrict__ outR   = out;
    int*    __restrict__ iterR  = iterOut;

    const int x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * w + x;

    if (x >= w || y >= h || idx >= w * h) return;
    if (!outR || !iterR || w <= 0 || h <= 0) return;

    const float scale = 1.0f / zoom;
    const float spanX = 3.5f * scale;
    const float spanY = spanX * (float)h / (float)w;

    const float2 c = pixelToComplex(x + 0.5f, y + 0.5f, w, h, spanX, spanY, offset);

    // --- Early-Out: analytisch + selektiv erweiterte „Schnauze“
    if (insideMainCardioidOrBulb(c.x, c.y)) {
        outR[idx]   = make_uchar4(0, 0, 0, 255);
        iterR[idx]  = maxIter;
        if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
            char msg[96]; int n = 0;
            n = luchs::d_append_str(msg, sizeof(msg), n, "[NOSE] early_inside (cardioid/p2) x=");
            n = luchs::d_append_int(msg, sizeof(msg), n, x);
            n = luchs::d_append_str(msg, sizeof(msg), n, " y=");
            n = luchs::d_append_int(msg, sizeof(msg), n, y);
            luchs::d_terminate(msg, sizeof(msg), n);
            LUCHS_LOG_DEVICE(msg);
        }
        return;
    } else {
        float sx, sy;
        if (likelyInteriorAfterFewSteps(c.x, c.y, sx, sy) &&
            insideHigherPeriodFromState(sx, sy, c.x, c.y))
        {
            outR[idx]   = make_uchar4(0, 0, 0, 255);
            iterR[idx]  = maxIter;
            if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
                char msg[96]; int n = 0;
                n = luchs::d_append_str(msg, sizeof(msg), n, "[NOSE] early_inside (p>=3) x=");
                n = luchs::d_append_int(msg, sizeof(msg), n, x);
                n = luchs::d_append_str(msg, sizeof(msg), n, " y=");
                n = luchs::d_append_int(msg, sizeof(msg), n, y);
                luchs::d_terminate(msg, sizeof(msg), n);
                LUCHS_LOG_DEVICE(msg);
            }
            return;
        }
    }

    float zx, zy;
    // 🐑 Schneefuchs: Warp-synchronisierte Iterationen (weniger Divergenz). (Bezug zu Schneefuchs)
    int it = mandelbrotIterations_warp(c.x, c.y, maxIter, zx, zy);

    const float3 rgb = colorFractalDetailed(c, zx, zy, it, maxIter);
    outR[idx] = make_uchar4(
        (unsigned char)(rgb.x * 255.0f),
        (unsigned char)(rgb.y * 255.0f),
        (unsigned char)(rgb.z * 255.0f),
        255
    );
    iterR[idx] = it;

    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        float norm = zx * zx + zy * zy;
        float t = (it < maxIter)
                    ? (((float)it + 1.0f - __log2f(__log2f(fmaxf(norm, 1.000001f)))) / (float)maxIter)
                    : 1.0f;
        float tClamped = fminf(fmaxf(t, 0.0f), 1.0f);

        char msg[192]; int n = 0;
        n = luchs::d_append_str(msg, sizeof(msg), n, "[KERNEL] x=");
        n = luchs::d_append_int(msg, sizeof(msg), n, x);
        n = luchs::d_append_str(msg, sizeof(msg), n, " y=");
        n = luchs::d_append_int(msg, sizeof(msg), n, y);
        n = luchs::d_append_str(msg, sizeof(msg), n, " it=");
        n = luchs::d_append_int(msg, sizeof(msg), n, it);
        n = luchs::d_append_str(msg, sizeof(msg), n, " tClamped=");
        n = luchs::d_append_float_fixed(msg, sizeof(msg), n, tClamped, 4);
        n = luchs::d_append_str(msg, sizeof(msg), n, " norm=");
        n = luchs::d_append_float_fixed(msg, sizeof(msg), n, norm, 4);
        luchs::d_terminate(msg, sizeof(msg), n);
        LUCHS_LOG_DEVICE(msg);
    }
}

// ---- ENTROPY-KERNEL ----
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

// ---- CONTRAST-KERNEL ----
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

// --- Host-Wrapper ------------------------------------------------------------

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

void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    // Otter: 32x8 bei performanceLogging – gute Occupancy/Coalescing. (Bezug zu Otter)
    // Hinweis: Für Tests kannst du 32x16 probieren; bei zu hohem Registerdruck ggf. zurück.
    dim3 block = Settings::performanceLogging ? dim3(32, 8) : dim3(16, 16);
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);

    auto t_launchStart = clk::now();
    mandelbrotKernel<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter);
    auto t_launchEnd = clk::now();

    auto t_syncStart = clk::now();
    cudaDeviceSynchronize();
    auto t_syncEnd = clk::now();

    auto t1 = clk::now();

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
