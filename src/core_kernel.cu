// 🐭 Maus-Kommentar: CUDA-Kernel für Mandelbrot-Fraktal und Entropieanalyse pro Tile
// - launch_mandelbrotHybrid: rendert Fraktalbild + Iterationen
// - computeTileEntropy: misst Entropie je Tile zur Bewertung der Bildstruktur (für Auto-Zoom)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "settings.hpp"
#include "core_kernel.h"

__device__ __forceinline__ uchar4 elegantColor(float t) {
    t = fmodf(t, 1.0f);  // t ∈ [0, 1)
    float s = sinf(3.14159f * t);  // sanfter Verlauf

    float r = 0.8f * s;
    float g = 0.5f + 0.4f * cosf(2.0f * 3.14159f * t);
    float b = 0.6f + 0.3f * sinf(4.0f * 3.14159f * t);

    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__device__ int mandelbrotIterations(float x0, float y0, int maxIter) {
    float x = 0.0f, y = 0.0f;
    int iter = 0;
    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        ++iter;
    }
    return iter;
}

__global__ void mandelbrotKernel(uchar4* output, int* iterationsOut,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float jx = (x - width  / 2.0f) / zoom + offset.x;
    float jy = (y - height / 2.0f) / zoom + offset.y;

    int iter = mandelbrotIterations(jx, jy, maxIterations);
    iterationsOut[y * width + x] = iter;

    float t = iter / (float)maxIterations;
    output[y * width + x] = elegantColor(t);
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

#ifdef DEBUG
        if (entropy > 3.0f) {
            printf("[ENTROPY] tile (%d,%d) idx %d -> H=%.4f\n",
                   tileX, tileY, tileIndex, entropy);
        }
#endif
    }
}

extern "C" void launch_mandelbrotHybrid(uchar4* output, int* d_iterations,
                                        int width, int height,
                                        float zoom, float2 offset,
                                        int maxIterations) {
    dim3 block(Settings::TILE_W, Settings::TILE_H);
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
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    dim3 grid(tilesX, tilesY);
    dim3 block(128);  // erhöhte Parallelität pro Tile

    entropyKernel<<<grid, block>>>(d_iterations, d_entropyOut,
                                   width, height,
                                   tileSize, maxIter);
    cudaDeviceSynchronize();
}
