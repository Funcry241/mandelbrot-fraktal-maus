// Datei: core_kernel.cu
// 🐭 Maus-Kommentar: CUDA-Kernel für Mandelbrot mit dynamischem Variance-Threshold (maximal komprimiert)

#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "settings.hpp"
#include "core_kernel.h"

__global__ void testKernel(uchar4* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) img[y * w + x] = make_uchar4((x * 255) / w, (y * 255) / h, 128, 255);
}

extern "C" void launch_debugGradient(uchar4* img, int w, int h) {
    dim3 t(Settings::TILE_W, Settings::TILE_H), b((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    printf("[INFO] DebugGradient Grid (%d, %d)\n", b.x, b.y);
    testKernel<<<b, t>>>(img, w, h);
    cudaDeviceSynchronize();
}

__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) return make_uchar4(0, 0, 0, 255);
    float log_zn = logf(zx * zx + zy * zy) * 0.5f, nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float t = (iter + 1.0f - nu) / maxIter, shift = fmodf(logf(zoom + 2.0f) * 0.07f, 1.0f);
    float r = powf(0.8f + 0.2f * cosf(6.28318f * (t + shift)), 1.5f);
    float g = powf(0.6f + 0.4f * cosf(6.28318f * (t + shift + 0.3f)), 1.5f);
    float b = powf(0.4f + 0.6f * cosf(6.28318f * (t + shift + 0.6f)), 1.5f);
    return make_uchar4(fminf(r * 255, 255), fminf(g * 255, 255), fminf(b * 255, 255), 255);
}

__global__ void mandelbrotHybrid(uchar4* img, int* iters, int w, int h, float zoom, float2 offset, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float cx = (x - w * 0.5f) / zoom + offset.x, cy = (y - h * 0.5f) / zoom + offset.y, zx = 0, zy = 0;
    int iter = 0;
    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt;
        ++iter;
    }
    img[y * w + x] = colorMap(iter, maxIter, zx, zy, zoom);
    iters[y * w + x] = iter;
}

extern "C" void launch_mandelbrotHybrid(uchar4* img, int* iters, int w, int h, float zoom, float2 offset, int maxIter) {
    static bool firstLaunch = true;
    dim3 t(Settings::TILE_W, Settings::TILE_H), b((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    if (firstLaunch) { printf("[INFO] Launch mandelbrotHybrid: Grid (%d, %d)\n", b.x, b.y); firstLaunch = false; }
    mandelbrotHybrid<<<b, t>>>(img, iters, w, h, zoom, offset, maxIter);
    cudaDeviceSynchronize();
}

__global__ void computeComplexity(const int* iters, int w, int h, float* comp, float threshold) {
    int tx = blockIdx.x, ty = blockIdx.y, x = tx * Settings::TILE_W + threadIdx.x, y = ty * Settings::TILE_H + threadIdx.y;
    int lid = threadIdx.y * blockDim.x + threadIdx.x, tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W;
    __shared__ float sum[Settings::TILE_W * Settings::TILE_H], sqSum[Settings::TILE_W * Settings::TILE_H];
    __shared__ int minIter[Settings::TILE_W * Settings::TILE_H], maxIter[Settings::TILE_W * Settings::TILE_H], count[Settings::TILE_W * Settings::TILE_H];
    float val = 0.0f; int valid = 0, iVal = 0;
    if (x < w && y < h) { iVal = iters[y * w + x]; val = (float)iVal; valid = 1; }
    sum[lid] = val; sqSum[lid] = val * val; minIter[lid] = maxIter[lid] = iVal; count[lid] = valid;
    __syncthreads();
    for (int s = (blockDim.x * blockDim.y) >> 1; s; s >>= 1) {
        if (lid < s) {
            sum[lid] += sum[lid + s]; sqSum[lid] += sqSum[lid + s];
            minIter[lid] = min(minIter[lid], minIter[lid + s]); maxIter[lid] = max(maxIter[lid], maxIter[lid + s]);
            count[lid] += count[lid + s];
        }
        __syncthreads();
    }
    if (!lid) {
        int n = count[0]; float score = 0.0f;
        if (n > 1) {
            float mean = sum[0] / n, var = (sqSum[0] / n) - mean * mean;
            int spread = maxIter[0] - minIter[0];
            score = var * (spread ? spread : 1);
        }
        comp[ty * tilesX + tx] = (score > threshold) ? score : 0.0f;
    }
}
