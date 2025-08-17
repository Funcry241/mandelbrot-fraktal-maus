/////
// MAUS: core kernel with tiny device formatter (no CRT redeclare; bounded; ASCII)
// 🐭 Maus: Feature „Schwarze Schnauze“ – Early-Out für Innenpunkte (Cardioid/Bulb).
// 🦦 Otter: Eye-Candy integriert (Smooth Coloring + Paletten via otter_color.hpp). (Bezug zu Otter)
// 🦊 Schneefuchs: Mathematisch exakt; Workload-Reduktion + deterministische ASCII-Logs. (Bezug zu Schneefuchs)
// 🐑 Schneefuchs: Warp-synchrones Escape & FMA – weniger Divergenz, weniger Instruktionen. (Bezug zu Schneefuchs)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include <chrono>
#include "common.hpp"
#include "luchs_device_format.hpp"   // <- tiny formatter (no snprintf)
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"
#include "otter_color.hpp"           // 🦦 Otter: Smooth Coloring + geschmackvolle Paletten (Bezug zu Otter)

// --- Tuning-Parameter --------------------------------------------------------
// Chunked Ballot: reduziert Sync-Overhead ohne Bildänderung.
namespace {
    // nur noch 1 Ballot pro 8 Iterationen
    constexpr int   WARP_CHUNK        = 8;

    // 🐑 Schneefuchs: Brent-Periodizität (robuster als seltene Punktprobe)
    // nach Anlaufphase wird ein "slow"-Punkt in verdoppelnden Abständen verglichen.
    constexpr int   BRENT_WARMUP      = 32;     // Iterationen vor Start der Zyklensuche
    // Epsilon^2 für „nahezu gleich“ (float, konservativ)
    constexpr float LOOP_EPS2         = 1e-8f;

    // 🦦 Otter: Standard-Look für Eye-Candy (Palette, Stripes, Gamma)
    constexpr otter::Palette kPalette   = otter::Palette::Glacier; // Aurora / Glacier / Ember
    constexpr float          kStripeF   = 3.0f;
    constexpr float          kStripeAmp = 0.10f;
    constexpr float          kGamma     = 2.2f;
}

// --- Helpers ----------------------------------------------------------------

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

// 🐑 Schneefuchs (neu): Warp-Iteration mit CHUNKED Ballot + Brent-Periodizität.
// Bildidentisch, weniger Divergenz. Innenpunkte enden deutlich früher.
__device__ __forceinline__ int mandelbrotIterations_warp(
    float cr, float ci, int maxIter, float& xr, float& xi)
{
    float x = 0.0f, y = 0.0f;
    int it = 0;

    // Brent: vergleiche gegen "slow"-Punkt in verdoppelnden Intervallen (power).
    float xs = 0.0f, ys = 0.0f; // slow
    int   power = 1;            // Vergleichsintervall (verdoppelt sich)
    int   lam   = 0;            // Schritte seit letztem Reset
    int   brent_hits = 0;       // zwei aufeinanderfolgende Treffer gefordert

    unsigned mask = 0xFFFFFFFFu;
#if (__CUDA_ARCH__ >= 700)
    mask = __activemask();
#endif
    bool active = true;

#pragma unroll 1
    for (int k = 0; k < maxIter; k += WARP_CHUNK) {

        // Innerer Block ohne Ballot – maximal WARP_CHUNK Schritte
#pragma unroll
        for (int s = 0; s < WARP_CHUNK; ++s) {
            if (!active) continue;

            float x2 = x * x;
            float y2 = y * y;
            if (x2 + y2 > 4.0f) { // entkommen
                active = false;
                continue;
            }

            // z = z^2 + c  (mit FMA)
            float xt = fmaf(x, x, -y2) + cr;     // x^2 - y^2 + cr
            y = fmaf(2.0f * x, y, ci);           // 2*x*y + ci
            x = xt;
            ++it;

            // Brent-Zyklensuche (nur nach Warmup, sehr günstig)
            if (it > BRENT_WARMUP) {
                float dx = x - xs, dy = y - ys;
                float d2 = dx * dx + dy * dy;
                if (d2 < LOOP_EPS2) {
                    if (++brent_hits >= 2) {
                        active = false;
                        it = maxIter;            // exakt "innen"
                        continue;
                    }
                } else {
                    brent_hits = 0;
                }
                if (++lam == power) {
                    xs = x; ys = y;             // slow nachziehen
                    power <<= 1;                // Intervall verdoppeln
                    lam = 0;
                }
            }
        }

        // Ein Warp-Vote pro CHUNK
        unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }

    xr = x; xi = y;
    return it;
}

// --- „Schwarze Schnauze“: Innenraum-Shortcut --------------------------------
// Otter: Early-Out für Punkte sicher in der Menge – spart komplette Iteration.
// Schneefuchs: Zwei exakte Tests (Hauptcardioide, period-2 Bulb).
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

// --- Kernel ------------------------------------------------------------------

// 🐑 Schneefuchs: __restrict__-Aliase helfen dem Compiler ohne API-Änderung.
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

    // 🐽 Schwarze Schnauze: Early-Out für Innenpunkte (schwarz, it=maxIter)
    if (insideMainCardioidOrBulb(c.x, c.y)) {
        outR[idx]   = make_uchar4(0, 0, 0, 255);
        iterR[idx]  = maxIter;
        if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
            char msg[96]; int n = 0;
            n = luchs::d_append_str(msg, sizeof(msg), n, "[NOSE] early_inside x=");
            n = luchs::d_append_int(msg, sizeof(msg), n, x);
            n = luchs::d_append_str(msg, sizeof(msg), n, " y=");
            n = luchs::d_append_int(msg, sizeof(msg), n, y);
            luchs::d_terminate(msg, sizeof(msg), n);
            LUCHS_LOG_DEVICE(msg);
        }
        return;
    }

    float zx, zy;
    // 🐑 Schneefuchs: Warp-Iteration in CHUNKs mit Brent-Periodizität.
    int it = mandelbrotIterations_warp(c.x, c.y, maxIter, zx, zy);

    // 🦦 Otter Eye-Candy: Smooth Coloring + Paletten (kein Banding)
    float3 col;
    if (it >= maxIter) {
        // Kern bleibt deterministisch dunkel (projektweit so gewünscht)
        col = make_float3(0.0f, 0.0f, 0.0f);
    } else {
        col = otter::shade(it, maxIter, zx, zy, kPalette, kStripeF, kStripeAmp, kGamma);
    }

    outR[idx] = make_uchar4(
        (unsigned char)(255.0f * fminf(fmaxf(col.x, 0.0f), 1.0f)),
        (unsigned char)(255.0f * fminf(fmaxf(col.y, 0.0f), 1.0f)),
        (unsigned char)(255.0f * fminf(fmaxf(col.z, 0.0f), 1.0f)),
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
