// Datei: src/core_kernel.cu
// Zeilen: 207
// 🐭 Maus-Kommentar: Mandelbrot-Kernel mit SUPERSAMPLING_COUNT & stabilisiertem Coloring. Verwendet log1p für präzise Norm-Werte & float-Epsilon-Korrektur. Keine grellen Farbsprünge mehr. Schneefuchs sagt: „Nur wer Zahlen mit Respekt behandelt, verdient ihr Licht.“

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"  // 🧮 Enthält SUPERSAMPLING_COUNT

// 🔧 Lokale Debug-Schalter
#define ENABLE_ENTROPY_LOGGING 0
constexpr float ENTROPY_LOG_THRESHOLD = 3.25f;
constexpr int LOG_TILE_MODULO = 32;

__device__ __forceinline__ uchar4 elegantColor(float t) {
    if (t < 0.0f) return make_uchar4(0, 0, 0, 255);
    float intensity = sqrtf(t);
    float r = 0.5f + 0.5f * __sinf(6.2831f * (intensity + 0.0f));
    float g = 0.5f + 0.5f * __sinf(6.2831f * (intensity + 0.33f));
    float b = 0.5f + 0.5f * __sinf(6.2831f * (intensity + 0.66f));
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__device__ int mandelbrotIterations(float x0, float y0, int maxIter, float& finalX, float& finalY) {
    float x = 0.0f, y = 0.0f;
    int iter = 0;
    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        ++iter;
    }
    finalX = x;
    finalY = y;
    return iter;
}

__global__ void mandelbrotKernel(uchar4* output, int* iterationsOut,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    constexpr int S = (SUPERSAMPLING_COUNT > 0) ? SUPERSAMPLING_COUNT : 1;
    const int SQRT_S = (int)sqrtf((float)S);

    float total = 0.0f;
    int iterSum = 0;

    for (int i = 0; i < SQRT_S; ++i) {
        for (int j = 0; j < SQRT_S; ++j) {
            float dx = (i + 0.5f) / SQRT_S;
            float dy = (j + 0.5f) / SQRT_S;

            float jx = (x + dx - width / 2.0f) / zoom + offset.x;
            float jy = (y + dy - height / 2.0f) / zoom + offset.y;

            float zx, zy;
            int iter = mandelbrotIterations(jx, jy, maxIterations, zx, zy);
            iterSum += iter;

            if (iter < maxIterations) {
                // 🧮 Verhindert log(0) durch minimale Epsilon-Korrektur
                float norm = fmaxf(zx * zx + zy * zy, 1e-12f);
                float t = (iter + 1.0f - log2f(log2f(norm))) / maxIterations;
                total += t;
            }
        }
    }

    float avg = total / (SQRT_S * SQRT_S);
    iterationsOut[y * width + x] = iterSum / (SQRT_S * SQRT_S);
    output[y * width + x] = elegantColor(avg);
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
            printf("[Entropy] Tile (%d,%d) idx %d -> H = %.4f\n", tileX, tileY, tileIndex, entropy);
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
        std::fprintf(stderr, "[FATAL] computeTileEntropy: Invalid input – tileSize=%d, width=%d, height=%d, maxIter=%d\n",
                     tileSize, width, height, maxIter);
        return;
    }

    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;

    if (tilesX == 0 || tilesY == 0) {
        std::fprintf(stderr, "[FATAL] computeTileEntropy: tile grid is zero-sized! tilesX=%d, tilesY=%d\n", tilesX, tilesY);
        return;
    }

    dim3 grid(tilesX, tilesY);
    dim3 block(128);

    entropyKernel<<<grid, block>>>(d_iterations, d_entropyOut,
                                   width, height,
                                   tileSize, maxIter);
    cudaDeviceSynchronize();
}
