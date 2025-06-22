// Datei: src/core_kernel.cu
// Zeilen: 159
// 🐭 Maus-Kommentar: Mandelbrot-Kernel mit Farbintelligenz – Sinusinterpolation mit Magnitude-Glühen. Debug-Entropie throttled. Schneefuchs: „Das Fraktal tanzt in Farben, wenn man es leise beobachtet.“

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"

// 🔧 Lokale Debug-Schalter (nicht global!)
#define ENABLE_ENTROPY_LOGGING 0
constexpr float ENTROPY_LOG_THRESHOLD = 3.25f;
constexpr int LOG_TILE_MODULO = 32; // Nur jedes 32. Tile loggen

__device__ __forceinline__ uchar4 elegantColor(float normIter, float mag) {
    if (normIter < 0.0f) return make_uchar4(0, 0, 0, 255);

    float glow = expf(-mag * 0.1f);
    float r = 0.6f + 0.4f * __sinf(3.0f + 5.0f * normIter + mag * 0.05f);
    float g = 0.5f + 0.5f * __sinf(2.0f + 4.0f * normIter + mag * 0.03f);
    float b = 0.4f + 0.6f * __sinf(4.0f + 6.0f * normIter + mag * 0.02f);
    r = my_clamp(r * glow, 0.0f, 1.0f);
    g = my_clamp(g * glow, 0.0f, 1.0f);
    b = my_clamp(b * glow, 0.0f, 1.0f);
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__device__ int mandelbrotIterations(float x0, float y0, int maxIter, float& magOut) {
    float x = 0.0f, y = 0.0f;
    int iter = 0;
    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        ++iter;
    }
    magOut = sqrtf(x * x + y * y);
    return iter;
}

__global__ void mandelbrotKernel(uchar4* output, int* iterationsOut,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float jx = (x - width / 2.0f) / zoom + offset.x;
    float jy = (y - height / 2.0f) / zoom + offset.y;

    float mag = 0.0f;
    int iter = mandelbrotIterations(jx, jy, maxIterations, mag);
    iterationsOut[y * width + x] = iter;

    float smoothIter = iter;
    if (iter < maxIterations) {
        float log_zn = logf(mag * mag) / 2.0f;
        float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
        smoothIter = iter + 1.0f - nu;
    }
    float normIter = smoothIter / maxIterations;
    output[y * width + x] = elegantColor(normIter, mag);
}

__global__ void entropyKernel(const int* iterations, float* entropyOut,
                              int width, int height, int tileSize,
                              int maxIter) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int startX = tileX * tileSize;
    int startY = tileY * tileSize;

    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        histo[i] = 0;
    __syncthreads();

    int localCount = 0;
    int tid = threadIdx.x;
    int threads = blockDim.x;
    int total = tileSize * tileSize;

    for (int idx = tid; idx < total; idx += threads) {
        int dx = idx % tileSize;
        int dy = idx / tileSize;
        int x = startX + dx;
        int y = startY + dy;
        if (x >= width || y >= height) continue;

        int iter = iterations[y * width + x];
        int bin = min(iter * 256 / (maxIter + 1), 255);
        atomicAdd(&histo[bin], 1);
        localCount++;
    }
    __syncthreads();

    __shared__ int totalCount;
    if (threadIdx.x == 0) totalCount = 0;
    __syncthreads();
    atomicAdd(&totalCount, localCount);
    __syncthreads();

    if (threadIdx.x == 0 && totalCount > 0) {
        float entropy = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float p = histo[i] / (float)totalCount;
            if (p > 0.0f)
                entropy -= p * log2f(p);
        }

        int tileIndex = tileY * gridDim.x + tileX;
        entropyOut[tileIndex] = entropy;

#if ENABLE_ENTROPY_LOGGING
        if (entropy > ENTROPY_LOG_THRESHOLD && tileIndex % LOG_TILE_MODULO == 0) {
            printf("[Entropy] Tile (%d,%d) idx %d -> H = %.4f\\n", tileX, tileY, tileIndex, entropy);
        }
#endif
    }
}

extern "C" void launch_mandelbrotHybrid(uchar4* output, int* d_iterations,
                                        int width, int height,
                                        float zoom, float2 offset,
                                        int maxIterations) {
    int tileSize = computeTileSizeFromZoom(zoom);
    dim3 block(tileSize, tileSize);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    mandelbrotKernel<<<grid, block>>>(output, d_iterations,
                                      width, height,
                                      zoom, offset,
                                      maxIterations);
    cudaDeviceSynchronize();
}

extern "C" void computeTileEntropy(const int* d_iterations,
                                   float* d_entropyOut,
                                   int width, int height,
                                   int tileSize,
                                   int maxIter) {
    if (tileSize <= 0 || width <= 0 || height <= 0 || maxIter <= 0) {
        std::fprintf(stderr, "[FATAL] computeTileEntropy: Invalid input – tileSize=%d, width=%d, height=%d, maxIter=%d\\n",
                     tileSize, width, height, maxIter);
        return;
    }

    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;

    if (tilesX == 0 || tilesY == 0) {
        std::fprintf(stderr, "[FATAL] computeTileEntropy: tile grid is zero-sized! tilesX=%d, tilesY=%d\\n", tilesX, tilesY);
        return;
    }

    dim3 grid(tilesX, tilesY);
    dim3 block(128);

    entropyKernel<<<grid, block>>>(d_iterations, d_entropyOut,
                                   width, height,
                                   tileSize, maxIter);
    cudaDeviceSynchronize();
}
