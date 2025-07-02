// Datei: src/core_kernel.cu
// Zeilen: 281
// 🐭 Maus-Kommentar: Mandelbrot-Kernel mit Supersampling und Entropie-Kontrastanalyse (Panda-Modul aktiv).
// Kombiniert: mandelbrotKernel, computeTileEntropy, contrastKernel, computeEntropyContrast.
// Schneefuchs sagt: „Kontrast schärft die Tiefe – und Panda sorgt für Klarheit.“
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"  // Enthält globale Parameter, falls nötig

#define ENABLE_ENTROPY_LOGGING 0
constexpr float ENTROPY_LOG_THRESHOLD [[maybe_unused]] = 3.25f;
constexpr int LOG_TILE_MODULO [[maybe_unused]] = 32;

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
                                 int maxIterations,
                                 int supersampling) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int S = supersampling;
    float totalColor = 0.0f;
    int validSamples = 0;
    int totalIter = 0;

    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            float dx = (i + 0.5f) / S;
            float dy = (j + 0.5f) / S;
            float jx = (x + dx - width / 2.0f) / zoom + offset.x;
            float jy = (y + dy - height / 2.0f) / zoom + offset.y;

            float zx, zy;
            int iter = mandelbrotIterations(jx, jy, maxIterations, zx, zy);
            totalIter += iter;

            if (iter < maxIterations) {
                float norm = zx * zx + zy * zy;
                float t = (iter + 1.0f - log2f(log2f(norm))) / maxIterations;
                totalColor += t;
                ++validSamples;
            }
        }
    }

    float avgColor = (validSamples > 0) ? (totalColor / validSamples) : -1.0f;
    int avgIter = totalIter / (S * S);

    if (Settings::debugGradient) {
        float val = (avgIter > 0) ? avgIter / (float)maxIterations : 0.0f;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        output[y * width + x] = make_uchar4(val * 255, val * 255, val * 255, 255);
        return;
    }

    output[y * width + x] = elegantColor(avgColor);
    iterationsOut[y * width + x] = avgIter;
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

__global__ void contrastKernel(const float* entropy, float* contrastOut,
                               int tilesX, int tilesY) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty * tilesX + tx;
    float center = entropy[idx];

    float sumDiff = 0.0f;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = tx + dx;
            int ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            int nIdx = ny * tilesX + nx;
            sumDiff += fabsf(entropy[nIdx] - center);
            count++;
        }
    }

    contrastOut[idx] = (count > 0) ? sumDiff / count : 0.0f;
}

extern "C" void launch_mandelbrotHybrid(uchar4* output, int* d_iterations,
                                        int width, int height,
                                        float zoom, float2 offset,
                                        int maxIterations,
                                        int supersampling) {
    int tileSize = computeTileSizeFromZoom(zoom);
    dim3 block(tileSize, tileSize);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    mandelbrotKernel<<<grid, block>>>(output, d_iterations,
                                      width, height,
                                      zoom, offset,
                                      maxIterations,
                                      supersampling);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

extern "C" void computeTileEntropy(const int* d_iterations,
                                   float* d_entropyOut,
                                   int width, int height,
                                   int tileSize,
                                   int maxIter) {
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;

    dim3 grid(tilesX, tilesY);
    dim3 block(128);

    entropyKernel<<<grid, block>>>(d_iterations, d_entropyOut,
                                   width, height,
                                   tileSize, maxIter);
    cudaDeviceSynchronize();
}

extern "C" void computeEntropyContrast(const int* d_iterations,
                                       float* d_entropyOut,
                                       float* d_contrastOut,
                                       int width, int height,
                                       int tileSize,
                                       int maxIter) {
    // 🐼 Entropie berechnen (wie zuvor)
    computeTileEntropy(d_iterations, d_entropyOut, width, height, tileSize, maxIter);

    // 🐼 Kontrast aus Entropie ableiten
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;

    dim3 grid((tilesX + 15) / 16, (tilesY + 15) / 16);
    dim3 block(16, 16);

    contrastKernel<<<grid, block>>>(d_entropyOut, d_contrastOut, tilesX, tilesY);
    cudaDeviceSynchronize();
}
