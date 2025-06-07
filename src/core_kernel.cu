// Datei: src/core_kernel.cu
#include <cstdio>
#include "settings.hpp"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include "core_kernel.h"

// 🐭 Gradient-Testbild
__global__ void testKernel(uchar4* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    img[y * w + x] = make_uchar4((x * 255) / w, (y * 255) / h, 128, 255);
}

extern "C" void launch_debugGradient(uchar4* img, int w, int h) {
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    printf("[INFO] DebugGradient Grid (%d,%d)\n", blocks.x, blocks.y);
    testKernel<<<blocks, threads>>>(img, w, h);
    cudaDeviceSynchronize();
}

// 🐭 Farbkodierung
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) return make_uchar4(0, 0, 0, 255);
    float log_zn = logf(zx * zx + zy * zy) * 0.5f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float t = fmodf(((iter + 1.0f - nu) / maxIter) * 3.0f, 1.0f);
    float hueShift = fmodf(logf(zoom + 1.0f) * 0.1f, 1.0f);
    float r = 0.5f + 0.5f * cosf(6.28318f * (t + hueShift + 0.0f));
    float g = 0.5f + 0.5f * cosf(6.28318f * (t + hueShift + 0.33f));
    float b = 0.5f + 0.5f * cosf(6.28318f * (t + hueShift + 0.67f));
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

// 🐭 Mandelbrot Kernel
__global__ void mandelbrotHybrid(uchar4* img, int* iterations, int w, int h, float zoom, float2 offset, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float cx = (x - w * 0.5f) / zoom + offset.x, cy = (y - h * 0.5f) / zoom + offset.y;
    float zx = 0.0f, zy = 0.0f; int iter = 0;
    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt; ++iter;
    }
    img[y * w + x] = colorMap(iter, maxIter, zx, zy, zoom);
    iterations[y * w + x] = iter;
}

extern "C" void launch_mandelbrotHybrid(uchar4* img, int* iterations, int w, int h, float zoom, float2 offset, int maxIter) {
    static bool firstLaunch = true;
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    if (firstLaunch) { printf("[INFO] Launch mandelbrotHybrid: Grid (%d,%d)\n", blocks.x, blocks.y); firstLaunch = false; }
    mandelbrotHybrid<<<blocks, threads>>>(img, iterations, w, h, zoom, offset, maxIter);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) printf("[ERROR] mandelbrotHybrid launch failed.\n");
}

// 🐭 Complexity Kernel
__global__ void computeComplexity(const int* iterations, int w, int h, float* complexity) {
    int tileX = blockIdx.x, tileY = blockIdx.y, tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W;
    int startX = tileX * Settings::TILE_W, startY = tileY * Settings::TILE_H;
    int localX = threadIdx.x, localY = threadIdx.y, x = startX + localX, y = startY + localY;
    int localId = localY * blockDim.x + localX;

    __shared__ float sum[Settings::TILE_W * Settings::TILE_H], sqSum[Settings::TILE_W * Settings::TILE_H];
    __shared__ int count[Settings::TILE_W * Settings::TILE_H];

    float value = 0.0f; int valid = 0;
    if (x < w && y < h) { value = static_cast<float>(iterations[y * w + x]); valid = 1; }
    sum[localId] = value; sqSum[localId] = value * value; count[localId] = valid;
    __syncthreads();

    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sum[localId] += sum[localId + stride];
            sqSum[localId] += sqSum[localId + stride];
            count[localId] += count[localId + stride];
        }
        __syncthreads();
    }

    if (localId == 0) {
        int n = count[0];
        float var = (n > 1) ? (sqSum[0] / n - (sum[0] / n) * (sum[0] / n)) : 0.0f;
        complexity[tileY * tilesX + tileX] = (var > Settings::VARIANCE_THRESHOLD) ? var : 0.0f;
    }
}
