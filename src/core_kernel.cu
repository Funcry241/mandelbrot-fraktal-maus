// Datei: src/core_kernel.cu
// 🐭 Maus-Kommentar: Alpha 64 - Supersampling vollständig entfernt. Klare Kernel-Signatur, Logging über Settings::debugLogging, deterministisch.
// 🦦 Otter: Vollständige Luchsifizierung - alle Logs über LUCHS_LOG_HOST/DEVICE. Keine Rohausgaben mehr.
// 🦊 Schneefuchs: Struktur bewahrt, keine Flüchtigkeit, keine faulen Tricks.

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
    float r = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.0f));
    float g = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.33f));
    float b = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.66f));
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

// ---- KOORDINATEN-MAPPING ----
__device__ __forceinline__ float2 pixelToComplex(float px, float py, int w, int h, float spanX, float spanY, float2 offset) {
    return make_float2((px / w - 0.5f) * spanX + offset.x,
                       (py / h - 0.5f) * spanY + offset.y);
}

// ---- MANDELBROT-ITERATION ----
__device__ int mandelbrotIterations(float x0, float y0, int maxIter, float& fx, float& fy) {
    float x = 0.0f, y = 0.0f; int i = 0;
    while (x * x + y * y <= 4.0f && i < maxIter) {
        float xt = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xt;
        ++i;
    }
    fx = x; fy = y;
    return i;
}

// ---- MANDELBROT-KERNEL ----
__global__ void mandelbrotKernel(
    uchar4* out, int* iterOut, int w, int h, float zoom, float2 offset, int maxIter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * w + x;

    if (x >= w || y >= h || idx >= w * h) return;

    float scale = 1.0f / zoom;
    float spanX = 3.5f * scale;
    float spanY = spanX * h / w;

    float2 c = pixelToComplex(x + 0.5f, y + 0.5f, w, h, spanX, spanY, offset);
    float zx, zy;
    int it = mandelbrotIterations(c.x, c.y, maxIter, zx, zy);

    float norm = zx * zx + zy * zy;
    float t = it - log2f(log2f(fmaxf(norm, 1.000001f)));
    t = fminf(fmaxf(t / maxIter, 0.0f), 1.0f);

    out[idx] = elegantColor(t);
    iterOut[idx] = it;

    if (Settings::debugLogging && threadIdx.x == 0 && threadIdx.y == 0) {
        LUCHS_LOG_DEVICE("BlockXY");
    }
}

// ---- ENTROPY & CONTRAST ----
__global__ void entropyKernel(const int* it, float* eOut, int w, int h, int tile, int maxIter) {
    int tX = blockIdx.x, tY = blockIdx.y, startX = tX * tile, startY = tY * tile;
    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) histo[i] = 0;
    __syncthreads();

    int total = tile * tile, tid = threadIdx.x, threads = blockDim.x;
    for (int idx = tid; idx < total; idx += threads) {
        int dx = idx % tile, dy = idx / tile, x = startX + dx, y = startY + dy;
        if (x >= w || y >= h) continue;
        int v = it[y * w + x]; v = max(0, v);
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

        if (Settings::debugLogging) {
            LUCHS_LOG_DEVICE("EntropyXY");
        }
    }
}

__global__ void contrastKernel(const float* e, float* cOut, int tilesX, int tilesY) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty * tilesX + tx;
    float center = e[idx], sum = 0.0f; int cnt = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = tx + dx, ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            int nIdx = ny * tilesX + nx;
            sum += fabsf(e[nIdx] - center); cnt++;
        }

    cOut[idx] = (cnt > 0) ? sum / cnt : 0.0f;

    if (Settings::debugLogging && threadIdx.x == 0 && threadIdx.y == 0) {
        LUCHS_LOG_DEVICE("ContrastXY");
    }
}

// ---- HOST-WRAPPER: Entropie & Kontrast ----
void computeCudaEntropyContrast(const int* d_it, float* d_e, float* d_c, int w, int h, int tile, int maxIter) {
    int tilesX = (w + tile - 1) / tile;
    int tilesY = (h + tile - 1) / tile;

    entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
    cudaDeviceSynchronize();

    contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16, 16)>>>(d_e, d_c, tilesX, tilesY);
    cudaDeviceSynchronize();
}

// ---- HOST-WRAPPER: Mandelbrot ----
void launch_mandelbrotHybrid(uchar4* out, int* d_it, int w, int h, float zoom, float2 offset, int maxIter, int tile) {
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[Kernel] %dx%d | Zoom: %.3e | Offset: (%.5f, %.5f) | Iter: %d | Tile: %d",
                       w, h, zoom, offset.x, offset.y, maxIter, tile);
    }

    if (out && d_it)
        mandelbrotKernel<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA ERROR] Kernel launch failed: %s", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    if (Settings::debugLogging) {
        int it[10] = { 0 };
        cudaMemcpy(it, d_it, sizeof(it), cudaMemcpyDeviceToHost);

        LUCHS_LOG_HOST("[Iter] First10: %d %d %d %d %d %d %d %d %d %d",
                       it[0], it[1], it[2], it[3], it[4], it[5], it[6], it[7], it[8], it[9]);
    }

    // 🦦 Otter: Luchsifizierung abgeschlossen - alles sauber, alles kontrolliert
}
