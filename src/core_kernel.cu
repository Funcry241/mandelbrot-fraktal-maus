// Datei: src/core_kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"

// ---- FARB-MAPPING ----
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

    out[idx] = elegantColor(tClamped);
    iterOut[idx] = it;

    if (doLog && isFirstThread) {
        LUCHS_LOG_DEVICE("mandelbrotKernel entered | x=%d y=%d idx=%d | c=(%.6f,%.6f) | it=%d norm=%.6f | %s %s %s %s",
            x, y, idx,
            c.x, c.y,
            it, norm,
            (it <= 2 ? "it<=2" : ""),
            (norm < 1.0f ? "norm<1" : ""),
            (t < 0.0f ? "t<0" : ""),
            (tClamped == 0 ? "tClamped=0" : ""));
    }
    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        LUCHS_LOG_DEVICE("block processed");
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

        if (doLog) {
            LUCHS_LOG_DEVICE("entropy computed");
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
    cOut[idx] = (cnt > 0) ? sum / cnt : 0.0f;

    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        LUCHS_LOG_DEVICE("contrast computed");
    }
}

// ---- HOST-WRAPPER: Entropie & Kontrast ----
void computeCudaEntropyContrast(
    const int* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter)
{
    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;
    entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
    cudaDeviceSynchronize();
    contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    cudaDeviceSynchronize();
}

// ---- HOST-WRAPPER: Mandelbrot ----
void launch_mandelbrotHybrid(
    uchar4* out, int* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int tile)
{
    dim3 block(16,16);
    dim3 grid((w + 15)/16, (h + 15)/16);

    if (Settings::debugLogging) {
        float scale = 1.0f / zoom;
        float spanX = 3.5f * scale;
        float spanY = spanX * h / w;

        float2 topLeft = {
            (-0.5f * spanX) + offset.x,
            (-0.5f * spanY) + offset.y
        };
        float2 bottomRight = {
            (0.5f * spanX) + offset.x,
            (0.5f * spanY) + offset.y
        };

        LUCHS_LOG_HOST(
            "[LAUNCH] %dx%d | Zoom=%.6f | Offset=(%.6f, %.6f) | Iter=%d | Tile=%d",
            w, h, zoom, offset.x, offset.y, maxIter, tile
        );
        LUCHS_LOG_HOST(
            "[VIEWPORT] spanX=%.6f spanY=%.6f",
            spanX, spanY
        );
        LUCHS_LOG_HOST(
            "[VIEWPORT] complex area: topLeft=(%.6f, %.6f) → bottomRight=(%.6f, %.6f)",
            topLeft.x, topLeft.y, bottomRight.x, bottomRight.y
        );
        LUCHS_LOG_HOST(
            "[THREADS] Launch blocks=%dx%d | blockSize=%dx%d | total threads=%d",
            grid.x, grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y
        );
    }

    if (!out || !d_it) {
        LUCHS_LOG_HOST("[FATAL] launch_mandelbrotHybrid aborted: null pointer");
        return;
    }

    mandelbrotKernel<<<grid,block>>>(out, d_it, w, h, zoom, offset, maxIter);
    LUCHS_LOG_HOST("[KERNEL] mandelbrotKernel<<<%d,%d>>> launched", grid.x, block.x);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA ERROR] Kernel launch failed: %s", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    if (Settings::debugLogging) {
        int sample[10] = {0};
        cudaMemcpy(sample, d_it, sizeof(sample), cudaMemcpyDeviceToHost);
        LUCHS_LOG_HOST(
            "[SAMPLE] Iteration: %d %d %d %d %d %d %d %d %d %d",
            sample[0], sample[1], sample[2], sample[3], sample[4],
            sample[5], sample[6], sample[7], sample[8], sample[9]
        );
    }
}
