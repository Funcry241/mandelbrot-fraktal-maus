// Datei: src/core_kernel.cu

#include <cstdio>
#include "settings.hpp"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include "core_kernel.h"

// 🐭 Test-Gradient-Kernel – IMMER definiert
__global__ void testKernel(uchar4* img, int width, int height) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height) return;

    unsigned char r = static_cast<unsigned char>((tx * 255) / width);
    unsigned char g = static_cast<unsigned char>((ty * 255) / height);
    img[ty * width + tx] = make_uchar4(r, g, 128, 255);
}

// 🐭 Debug-Gradient Launcher
extern "C" void launch_debugGradient(
    uchar4* img,
    int width, int height
) {
    int bx = (width  + Settings::TILE_W - 1) / Settings::TILE_W;
    int by = (height + Settings::TILE_H - 1) / Settings::TILE_H;
    dim3 blocks(bx, by);
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    printf("[INFO] Starte DebugGradient mit Grid (%d,%d)\n", blocks.x, blocks.y);
    testKernel<<<blocks, threads>>>(img, width, height);
    cudaDeviceSynchronize();
}

// 🐭 Farbkodierung für Mandelbrot
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) {
        return make_uchar4(0, 0, 0, 255);
    }

    float log_zn = logf(zx * zx + zy * zy) / 2.0f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float smoothIter = iter + 1.0f - nu;

    float t = smoothIter / maxIter;
    t = fmodf(t * 3.0f, 1.0f);

    float hueShift = fmodf(logf(zoom + 1.0f) * 0.1f, 1.0f);

    float r = 0.5f + 0.5f * cosf(6.28318f * (t + hueShift + 0.0f));
    float g = 0.5f + 0.5f * cosf(6.28318f * (t + hueShift + 0.33f));
    float b = 0.5f + 0.5f * cosf(6.28318f * (t + hueShift + 0.67f));

    return make_uchar4(
        static_cast<unsigned char>(r * 255.0f),
        static_cast<unsigned char>(g * 255.0f),
        static_cast<unsigned char>(b * 255.0f),
        255
    );
}

// 🐭 Mandelbrot Haupt-Kernel – schreibt jetzt Iterationen separat
__global__ void mandelbrotHybrid(
    uchar4* img,
    int* iterations,
    int width, int height,
    float zoom, float2 offset,
    int maxIter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float cx = (static_cast<float>(x) - width * 0.5f) / zoom + offset.x;
    float cy = (static_cast<float>(y) - height * 0.5f) / zoom + offset.y;
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;

    const float escapeRadius2 = 4.0f;
    while (zx * zx + zy * zy < escapeRadius2 && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt;
        ++iter;
    }

    img[y * width + x] = colorMap(iter, maxIter, zx, zy, zoom);
    iterations[y * width + x] = iter;
}

// 🐭 Wrapper für Hauptkernel
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,
    int* iterations,
    int width, int height,
    float zoom, float2 offset,
    int maxIter
) {
    static bool firstLaunch = true;

    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    if (firstLaunch) {
        printf("[INFO] Launching mandelbrotHybrid: Grid (%d,%d), Threads (%d,%d)\n", blocks.x, blocks.y, threads.x, threads.y);
        firstLaunch = false;
    }

    mandelbrotHybrid<<<blocks, threads>>>(img, iterations, width, height, zoom, offset, maxIter);

    cudaError_t errSync  = cudaDeviceSynchronize();
    cudaError_t errAsync = cudaGetLastError();
    if (errSync != cudaSuccess) {
        printf("[SYNC ERROR] mandelbrotHybrid: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess) {
        printf("[ASYNC ERROR] mandelbrotHybrid: %s\n", cudaGetErrorString(errAsync));
    }
}

// 🐭 Complexity Kernel – auf Iterationspuffer
__global__ void computeComplexity(
    const int* iterations,
    int width,
    int height,
    float* complexity
) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tilesX = (width + Settings::TILE_W - 1) / Settings::TILE_W;

    int startX = tileX * Settings::TILE_W;
    int startY = tileY * Settings::TILE_H;

    int localX = threadIdx.x;
    int localY = threadIdx.y;

    int x = startX + localX;
    int y = startY + localY;

    const int localId = localY * blockDim.x + localX;

    __shared__ float sharedSum[Settings::TILE_W * Settings::TILE_H];
    __shared__ float sharedSqSum[Settings::TILE_W * Settings::TILE_H];
    __shared__ int   sharedCount[Settings::TILE_W * Settings::TILE_H];

    float value = 0.0f;
    int valid = 0;

    if (x < width && y < height) {
        int iter = iterations[y * width + x];
        value = static_cast<float>(iter);
        valid = 1;
    }

    sharedSum[localId] = value;
    sharedSqSum[localId] = value * value;
    sharedCount[localId] = valid;
    __syncthreads();

    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedSum[localId] += sharedSum[localId + stride];
            sharedSqSum[localId] += sharedSqSum[localId + stride];
            sharedCount[localId] += sharedCount[localId + stride];
        }
        __syncthreads();
    }

    if (localId == 0) {
        int count = sharedCount[0];
        if (count > 1) {
            float mean = sharedSum[0] / count;
            float meanSq = sharedSqSum[0] / count;
            float variance = meanSq - mean * mean;
            complexity[tileY * tilesX + tileX] = (variance > 1e-6f) ? variance : 0.0f;
        } else {
            complexity[tileY * tilesX + tileX] = 0.0f;
        }
    }
}
