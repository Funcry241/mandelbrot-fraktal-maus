// Datei: core_kernel.cu
// 🐭 Maus-Kommentar: CUDA-Kernel für Mandelbrot mit dynamischem Variance-Threshold (perfektioniert)

#include <cstdio>
#include "settings.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core_kernel.h"

// 🐭 Dynamischer Threshold als Device-Symbol
__device__ float deviceVarianceThreshold = 1e-6f;

// 🐭 Gradient-Testkernel
__global__ void testKernel(uchar4* __restrict__ img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
        img[y * w + x] = make_uchar4((x * 255) / w, (y * 255) / h, 128, 255);
}

extern "C" void launch_debugGradient(uchar4* img, int w, int h) {
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    printf("[INFO] DebugGradient Grid (%d, %d)\n", blocks.x, blocks.y);
    testKernel<<<blocks, threads>>>(img, w, h);
    if (cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess)
        fprintf(stderr, "[ERROR] launch_debugGradient failed: %s\n", cudaGetErrorString(err));
}

// 🐭 Farbzuordnung mit sanftem Verlauf
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) return make_uchar4(0, 0, 0, 255);

    float log_zn = logf(zx * zx + zy * zy) * 0.5f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float t = (iter + 1.0f - nu) / maxIter;
    float shift = fmodf(logf(zoom + 2.0f) * 0.07f, 1.0f);

    float r = powf(0.8f + 0.2f * cosf(6.28318f * (t + shift)), 1.5f);
    float g = powf(0.6f + 0.4f * cosf(6.28318f * (t + shift + 0.3f)), 1.5f);
    float b = powf(0.4f + 0.6f * cosf(6.28318f * (t + shift + 0.6f)), 1.5f);

    return make_uchar4(fminf(r * 255.0f, 255.0f), fminf(g * 255.0f, 255.0f), fminf(b * 255.0f, 255.0f), 255);
}

// 🐭 Mandelbrot-Rendering mit Iterationspuffer
__global__ void mandelbrotHybrid(uchar4* __restrict__ img, int* __restrict__ iterations, int w, int h, float zoom, float2 offset, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float cx = (x - w * 0.5f) / zoom + offset.x;
    float cy = (y - h * 0.5f) / zoom + offset.y;
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt;
        ++iter;
    }
    img[y * w + x] = colorMap(iter, maxIter, zx, zy, zoom);
    iterations[y * w + x] = iter;
}

extern "C" void launch_mandelbrotHybrid(uchar4* img, int* iterations, int w, int h, float zoom, float2 offset, int maxIter) {
    static bool firstLaunch = true;
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    if (firstLaunch) {
        printf("[INFO] Launch mandelbrotHybrid: Grid (%d, %d)\n", blocks.x, blocks.y);
        firstLaunch = false;
    }
    mandelbrotHybrid<<<blocks, threads>>>(img, iterations, w, h, zoom, offset, maxIter);
    if (cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess)
        fprintf(stderr, "[ERROR] launch_mandelbrotHybrid failed: %s\n", cudaGetErrorString(err));
}

// 🐭 Komplexitätsberechnung mit dynamischem Threshold
__global__ void computeComplexity(const int* __restrict__ iterations, int w, int h, float* __restrict__ complexity) {
    int tileX = blockIdx.x, tileY = blockIdx.y;
    int startX = tileX * Settings::TILE_W;
    int startY = tileY * Settings::TILE_H;
    int x = startX + threadIdx.x;
    int y = startY + threadIdx.y;
    int lid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float sum[Settings::TILE_W * Settings::TILE_H];
    __shared__ float sqSum[Settings::TILE_W * Settings::TILE_H];
    __shared__ int minIter[Settings::TILE_W * Settings::TILE_H];
    __shared__ int maxIter[Settings::TILE_W * Settings::TILE_H];
    __shared__ int count[Settings::TILE_W * Settings::TILE_H];

    float val = 0.0f;
    int valid = 0, iterVal = 0;

    if (x < w && y < h) {
        iterVal = iterations[y * w + x];
        val = static_cast<float>(iterVal);
        valid = 1;
    }

    sum[lid] = val;
    sqSum[lid] = val * val;
    minIter[lid] = iterVal;
    maxIter[lid] = iterVal;
    count[lid] = valid;
    __syncthreads();

    // 🐭 Reduction für Summe, Quadrat-Summe, Minimum, Maximum und Count
    for (int stride = (blockDim.x * blockDim.y) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum[lid] += sum[lid + stride];
            sqSum[lid] += sqSum[lid + stride];
            minIter[lid] = min(minIter[lid], minIter[lid + stride]);
            maxIter[lid] = max(maxIter[lid], maxIter[lid + stride]);
            count[lid] += count[lid + stride];
        }
        __syncthreads();
    }

    if (lid == 0) {
        int n = count[0];
        float score = 0.0f;
        if (n > 1) {
            float mean = sum[0] / n;
            float var = (sqSum[0] / n) - (mean * mean);
            int spread = maxIter[0] - minIter[0];
            score = var * (spread > 0 ? spread : 1);
        }
        int tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W;
        complexity[tileY * tilesX + tileX] = (score > deviceVarianceThreshold) ? score : 0.0f;
    }
}
