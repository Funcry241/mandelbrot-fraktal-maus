// 🐭 Maus: Farb-Upgrade – Continuous Escape Time + feine „Stripe“-Details.
// 🦦 Otter: Nur diese Datei geändert, keine Signaturen angerührt. Sanft, detailreich, deterministisch. (Bezug zu Otter)
// 🦊 Schneefuchs: Keine externen Abhängigkeiten, nur lokale Helfer. Logs ASCII, unverändert. (Bezug zu Schneefuchs)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include <chrono> // Otter: für Zeitmessung
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"

#ifdef __CUDA_ARCH__
__device__ int sprintf(char* str, const char* format, ...);
#endif

// 🐭 fract ersetzt – CUDA kennt kein GLSL
__device__ __forceinline__ float fract(float x) {
    return x - floorf(x); // Schneefuchs: wie GLSL, aber CUDA-eigen
}

// ---- FARB-MAPPING (legacy) ----
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

// ---- KOORDINATEN-MAPPING ----
__device__ __forceinline__ float2 pixelToComplex(
    float px, float py, int w, int h,
    float spanX, float spanY, float2 offset)
{
    return make_float2(
        (px / w - 0.5f) * spanX + offset.x,
        (py / h - 0.5f) * spanY + offset.y
    );
}

// ---- MANDELBROT-ITERATION ----
__device__ int mandelbrotIterations(
    float x0, float y0, int maxIter,
    float& fx, float& fy)
{
    float x = 0.0f, y = 0.0f;
    int i = 0;
    while (x * x + y * y <= 4.0f && i < maxIter) {
        float xt = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xt;
        ++i;
    }
    fx = x;
    fy = y;
    return i;
}

// ---- RÜSSELWARZE: HSV-Konvertierung & Strukturchaos ----
__device__ float3 hsvToRgb(float h, float s, float v) {
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
        case 5: r = v, g = p, b = q; break;
    }
    return make_float3(r, g, b);
}

__device__ float pseudoRandomWarze(float x, float y) {
    float r = sqrtf(x * x + y * y);
    float angle = atan2f(y, x);
    return 0.5f + 0.5f * __sinf(r * 6.0f + angle * 4.0f);
}

// ── NEU: Continuous Escape-Time (CEC) + „Stripe“-Details ────────────────────
__device__ __forceinline__
void computeCEC(float zx, float zy, int it, int maxIt, float& nu, float& stripe)
{
    float norm = zx * zx + zy * zy;
    if (it >= maxIt) {
        nu = 1.0f;
        stripe = 0.0f;
        return;
    }
    float mu = (float)it + 1.0f - log2f(log2f(fmaxf(norm, 1.000001f)));
    nu = fminf(fmaxf(mu / (float)maxIt, 0.0f), 1.0f);
    float frac = fract(mu);
    stripe = powf(0.5f + 0.5f * __sinf(6.2831853f * frac), 0.75f);
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
    float hue   = fract(nu * 0.25f + angle * 0.08f * 0.15915494f);
    float val = 0.3f + 0.7f * stripe;
    float sat = 0.9f;
    return hsvToRgb(hue, sat, val);
}

// ---- MANDELBROT-KERNEL ----
__global__ void mandelbrotKernel(
    uchar4* out, int* iterOut,
    int w, int h, float zoom, float2 offset, int maxIter)
{
    const bool doLog = Settings::debugLogging;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * w + x;

    if (x >= w || y >= h || idx >= w * h) return;
    if (!out || !iterOut || w <= 0 || h <= 0) return;

    float scale = 1.0f / zoom;
    float spanX = 3.5f * scale;
    float spanY = spanX * h / w;

    float2 c = pixelToComplex(
        x + 0.5f, y + 0.5f,
        w, h, spanX, spanY, offset
    );
    float zx, zy;
    int it = mandelbrotIterations(c.x, c.y, maxIter, zx, zy);

    float3 rgb = colorFractalDetailed(c, zx, zy, it, maxIter);

    unsigned char rC = static_cast<unsigned char>(rgb.x * 255.0f);
    unsigned char gC = static_cast<unsigned char>(rgb.y * 255.0f);
    unsigned char bC = static_cast<unsigned char>(rgb.z * 255.0f);
    out[idx] = make_uchar4(rC, gC, bC, 255);
    iterOut[idx] = it;

    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        float norm = zx * zx + zy * zy;
        float t = (it < maxIter) ? (((float)it + 1.0f - log2f(log2f(fmaxf(norm, 1.000001f)))) / (float)maxIter)
                                 : 1.0f;
        float tClamped = fminf(fmaxf(t, 0.0f), 1.0f);
        char msg[256];
        int n = 0;
        n += sprintf(msg + n, "[KERNEL] x=%d y=%d it=%d ", x, y, it);
        n += sprintf(msg + n, "tClamped=%.4f norm=%.4f ", tClamped, norm);
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
        char msg[256];
        sprintf(msg,
            "[ENTROPY-DEBUG] tX=%d tY=%d tile=%d w=%d h=%d tilesX=%d tilesY=%d tileIndex=%d",
            tX, tY, tile, w, h, tilesX, tilesY, tileIndex);
        LUCHS_LOG_DEVICE(msg);
    }

    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        histo[i] = 0;
    __syncthreads();

    int total = tile * tile;
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
            if (p > 0.0f) entropy -= p * log2f(p);
        }
        eOut[tileIndex] = entropy;

        if (doLog) {
            char msg[128];
            sprintf(msg, "[ENTROPY] tile=(%d,%d) entropy=%.5f", tX, tY, entropy);
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
        char msg[128];
        int n = 0;
        n += sprintf(msg + n, "[CONTRAST] tile=(%d,%d) contrast=%.5f", tx, ty, contrast);
        LUCHS_LOG_DEVICE(msg);
    }
}

// ---- HOST-WRAPPER: Entropie & Kontrast ----
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

// ---- HOST-WRAPPER: Mandelbrot ----
void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int tile)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    dim3 block(16,16);
    dim3 grid((w + 15)/16, (h + 15)/16);

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
