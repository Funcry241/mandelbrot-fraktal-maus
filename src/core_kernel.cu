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

// ---- MANDELBROT-KERNEL ----
__global__ void mandelbrotKernel(
    uchar4* out, int* iterOut,
    int w, int h, float zoom, float2 offset, int maxIter)
{
    const bool doLog = Settings::debugLogging;
    const bool isFirstThread =
        blockIdx.x == 0 && blockIdx.y == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0;

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

    float norm = zx * zx + zy * zy;
    float t = it - log2f(log2f(fmaxf(norm, 1.000001f)));
    float tClamped = fminf(fmaxf(t / maxIter, 0.0f), 1.0f);

    // 🐽 RÜSSELWARZE-FARBREGELUNG
    float3 rgb;
    if (it == maxIter) {
        rgb = make_float3(0.05f, 0.05f, 0.08f);
    } else {
        float r = sqrtf(c.x * c.x + c.y * c.y);
        float angle = atan2f(c.y, c.x);
        float h = fract(angle / (2.0f * CUDART_PI_F));
        float v = 0.3f + 0.5f * pseudoRandomWarze(c.x, c.y);
        rgb = hsvToRgb(h, 0.85f, v);
    }
    unsigned char rC = static_cast<unsigned char>(rgb.x * 255.0f);
    unsigned char gC = static_cast<unsigned char>(rgb.y * 255.0f);
    unsigned char bC = static_cast<unsigned char>(rgb.z * 255.0f);
    out[idx] = make_uchar4(rC, gC, bC, 255);
    iterOut[idx] = it;

    // 🐭 Logging: nur 1 Thread pro Block
    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        char msg[512];
        int n = 0;
        n += sprintf(msg + n, "[KERNEL] x=%d y=%d it=%d ", x, y, it);
        n += sprintf(msg + n, "tClamped=%.4f (max=%d) norm=%.4f ", tClamped, maxIter, norm);
        n += sprintf(msg + n, "| center=(%.5f, %.5f)", c.x, c.y);
        LUCHS_LOG_DEVICE(msg);

        if (it == maxIter) {
            LUCHS_LOG_DEVICE("[KERNEL] Pixel inside Mandelbrot set (maxIter reached)");
        }
        if (tClamped == 0.0f) {
            LUCHS_LOG_DEVICE("[KERNEL] WARNING: tClamped = 0 – possible loss of gradient");
        }
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
        int tilesX = (w + tile - 1) / tile;
        int tileIndex = tY * tilesX + tX;
        eOut[tileIndex] = entropy;

        // 🐭 Logging: gezielt Histogramm prüfen
        if (doLog) {
            char msg[256];
            int n = 0;
            n += sprintf(msg + n, "[ENTROPY] tile=(%d,%d) ", tX, tY);
            n += sprintf(msg + n, "entropy=%.5f ", entropy);
            n += sprintf(msg + n, "| histo[0]=%d histo[255]=%d", histo[0], histo[255]);
            LUCHS_LOG_DEVICE(msg);

            if (entropy < 0.01f) {
                LUCHS_LOG_DEVICE("[ENTROPY] WARNING: Entropy ≈ 0 – likely uniform region");
            }
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

    // 🐭 Logging: erster Thread jeder Blockgruppe meldet Kontrastwerte
    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        char msg[256];
        int n = 0;
        n += sprintf(msg + n, "[CONTRAST] tile=(%d,%d) ", tx, ty);
        n += sprintf(msg + n, "index=%d ", idx);
        n += sprintf(msg + n, "| center=%.5f ", center);
        n += sprintf(msg + n, "| contrast=%.5f", contrast);
        LUCHS_LOG_DEVICE(msg);

        if (contrast == 0.0f) {
            LUCHS_LOG_DEVICE("[CONTRAST] WARNING: contrast = 0 → no neighborhood variation");
        }
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

    // 🧽 Otter: Entropiespeicher vorher nullen, damit leer korrekt leer bleibt
    cudaMemset(d_e, 0, tilesX * tilesY * sizeof(float));

    entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
    cudaDeviceSynchronize();

    auto mid = clk::now();

    contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    cudaDeviceSynchronize();

    auto end = clk::now();

    if (Settings::debugLogging) {
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

    // 🦦 Otter: Timing für Kernel-Launch
    auto t_launchStart = clk::now();
    mandelbrotKernel<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter);
    auto t_launchEnd = clk::now();

    // 🐑 Schneefuchs: Timing für Device-Synchronisation
    auto t_syncStart = clk::now();
    cudaDeviceSynchronize();
    auto t_syncEnd = clk::now();

    // 🦦 Otter: Gesamtzeit
    auto t1 = clk::now();

    if (Settings::debugLogging) {
        double launchMs = std::chrono::duration<double, std::milli>(t_launchEnd - t_launchStart).count();
        double syncMs   = std::chrono::duration<double, std::milli>(t_syncEnd - t_syncStart).count();
        double totalMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LUCHS_LOG_HOST("[TIME] Mandelbrot | Launch %.3f ms | Sync %.3f ms | Total %.3f ms", launchMs, syncMs, totalMs);
    }
}
