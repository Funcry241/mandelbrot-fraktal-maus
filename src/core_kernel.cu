// Datei: src/core_kernel.cu
// 🐭 Maus-Kommentar: Logging-Zustände jetzt explizit kontrolliert. Kein Spam, klare Zustände.
// 🦦 Otter: DebugLogging respektiert – keine unnötige Ausgabe. Alle Pfade eindeutig.
// 🦊 Schneefuchs: Struktur erhalten, keine Ausgabe ohne Anlass, Clarity bewahrt.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_device.hpp"
#include "luchs_log_host.hpp"

__device__ __forceinline__ uchar4 elegantColor(float t) {
    t = sqrtf(fminf(fmaxf(t, 0.0f), 1.0f));
    float r = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.0f));
    float g = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.33f));
    float b = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.66f));
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__device__ __forceinline__ float2 pixelToComplex(float px, float py, int w, int h, float spanX, float spanY, float2 offset) {
    return make_float2((px / w - 0.5f) * spanX + offset.x,
                       (py / h - 0.5f) * spanY + offset.y);
}

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

__global__ void mandelbrotKernel(
    uchar4* out, int* iterOut, int w, int h, float zoom, float2 offset, int maxIter)
{
    const bool doLog = Settings::debugLogging;
    const bool isFirstThread = (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0);

    if (doLog && isFirstThread) {
        LUCHS_LOG_DEVICE("[KERNEL] mandelbrotKernel entered");
    }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * w + x;

    if (x >= w || y >= h || idx >= w * h) return;
    if (!out || !iterOut || w <= 0 || h <= 0) return;

    float scale = 1.0f / zoom;
    float spanX = 3.5f * scale;
    float spanY = spanX * h / w;

    float2 c = pixelToComplex(x + 0.5f, y + 0.5f, w, h, spanX, spanY, offset);
    float zx, zy;
    int it = mandelbrotIterations(c.x, c.y, maxIter, zx, zy);

    float norm = zx * zx + zy * zy;
    float t = it - log2f(log2f(fmaxf(norm, 1.000001f)));
    float tClamped = fminf(fmaxf(t / maxIter, 0.0f), 1.0f);

    out[idx] = elegantColor(tClamped);
    iterOut[idx] = it;

    if (doLog && isFirstThread) {
        if (it <= 2)        LUCHS_LOG_DEVICE("[WRITE] it <= 2");
        if (norm < 1.0f)    LUCHS_LOG_DEVICE("[MAP] norm < 1.0");
        if (t < 0.0f)       LUCHS_LOG_DEVICE("[MAP] t < 0");
        if (tClamped == 0)  LUCHS_LOG_DEVICE("[MAP] tClamped == 0");
    }

    if (doLog && threadIdx.x == 0 && threadIdx.y == 0) {
        LUCHS_LOG_DEVICE("[KERNEL] block (%d,%d) done", blockIdx.x, blockIdx.y);
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
            LUCHS_LOG_DEVICE("EntropyKernel: entropy computed");
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
        LUCHS_LOG_DEVICE("ContrastKernel: contrast computed");
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

        cudaPointerAttributes attr;
        cudaError_t attrErr = cudaPointerGetAttributes(&attr, out);
        LUCHS_LOG_HOST("[DEBUG] cudaPointerGetAttributes(out): err=%d type=%d", attrErr, attr.type);
    }

    if (out && d_it) {
        mandelbrotKernel<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter);
        LUCHS_LOG_HOST("[KERNEL] mandelbrotKernel<<<%d,%d>>> launched", grid.x, block.x);
    } else {
        LUCHS_LOG_HOST("[FATAL] launch_mandelbrotHybrid aborted: null device pointer(s)");
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA ERROR] Kernel launch failed: %s", cudaGetErrorString(err));
    } else {
        LUCHS_LOG_HOST("[CHECK] cudaGetLastError returned success");
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA ERROR] Kernel sync failed: %s", cudaGetErrorString(syncErr));
        return;
    }

    if (Settings::debugLogging) {
        int it[10] = { 0 };
        cudaMemcpy(it, d_it, sizeof(it), cudaMemcpyDeviceToHost);
        LUCHS_LOG_HOST("[Mandelbrot] Iteration sample: %d %d %d %d %d %d %d %d %d %d",
                       it[0], it[1], it[2], it[3], it[4], it[5], it[6], it[7], it[8], it[9]);
    }
}
