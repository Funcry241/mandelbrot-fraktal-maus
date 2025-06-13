// Datei: src/core_kernel.cu
// 🐭 Maus-Kommentar: Mandelbrot-Kernel mit Farbverlauf + Komplexitätsanalyse je Tile

#include <cstdio>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include "settings.hpp"
#include "core_kernel.h"

// 🎯 Variance-Schwelle für Auto-Zoom (vom Host gesetzt, auf Device genutzt)
__device__ float deviceVarianceThreshold = 1e-6f;

extern "C" void setDeviceVarianceThreshold(float threshold) {
    cudaMemcpyToSymbol(deviceVarianceThreshold, &threshold, sizeof(float));
}

// 🌈 Testkernel: einfacher RGB-Verlauf
__global__ void testKernel(uchar4* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    img[y * w + x] = make_uchar4((x * 255) / w, (y * 255) / h, 128, 255);
}

extern "C" void launch_debugGradient(uchar4* img, int w, int h, float zoom) {
    (void)zoom;
    dim3 threads(Settings::BASE_TILE_SIZE, Settings::BASE_TILE_SIZE);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    printf("[INFO] DebugGradient Grid (%d, %d)\n", blocks.x, blocks.y);
    testKernel<<<blocks, threads>>>(img, w, h);
    cudaDeviceSynchronize();
}

// 🎨 Smooth Color Mapping
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) return make_uchar4(0, 0, 0, 255);
    float log_zn = logf(zx * zx + zy * zy) * 0.5f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float t = (iter + 1.0f - nu) / maxIter;
    float zoomShift = fmodf(logf(zoom + 2.0f) * 0.07f, 1.0f);

    float r = powf(0.8f + 0.2f * cosf(6.28318f * (t + zoomShift)), 1.5f);
    float g = powf(0.6f + 0.4f * cosf(6.28318f * (t + zoomShift + 0.3f)), 1.5f);
    float b = powf(0.4f + 0.6f * cosf(6.28318f * (t + zoomShift + 0.6f)), 1.5f);
    return make_uchar4(fminf(r * 255, 255), fminf(g * 255, 255), fminf(b * 255, 255), 255);
}

// 🌀 Hybrid-Renderer: schreibt Farbe + Iterationen
__global__ void mandelbrotHybrid(uchar4* img, int* iterations, int w, int h, float zoom, float2 offset, int maxIter) {
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
    dim3 threads(Settings::BASE_TILE_SIZE, Settings::BASE_TILE_SIZE);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    if (firstLaunch) {
        printf("[INFO] Launch mandelbrotHybrid: Grid (%d, %d)\n", blocks.x, blocks.y);
        firstLaunch = false;
    }

    mandelbrotHybrid<<<blocks, threads>>>(img, iterations, w, h, zoom, offset, maxIter);
    cudaDeviceSynchronize();
}

// 🧠 Komplexitätsbewertung (pro Tile: Standardabweichung)
__global__ void computeComplexityKernel(
    const int* iterations,
    int w,
    int h,
    float* complexity,
    int tileSize
) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tilesX = (w + tileSize - 1) / tileSize;

    int startX = tileX * tileSize;
    int startY = tileY * tileSize;

    int localX = threadIdx.x;
    int localY = threadIdx.y;
    int x = startX + localX;
    int y = startY + localY;

    int localId = localY * blockDim.x + localX;

    extern __shared__ float sharedData[];
    float* sumIter   = sharedData;
    float* sumIterSq = sharedData + blockDim.x * blockDim.y;
    int*   count     = (int*)(sharedData + 2 * blockDim.x * blockDim.y);

    float iterValue = 0.0f;
    int valid = 0;

    if (x < w && y < h) {
        iterValue = (float)iterations[y * w + x];
        valid = 1;
    }

    sumIter[localId] = iterValue;
    sumIterSq[localId] = iterValue * iterValue;
    count[localId] = valid;

    __syncthreads();

    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sumIter[localId] += sumIter[localId + stride];
            sumIterSq[localId] += sumIterSq[localId + stride];
            count[localId] += count[localId + stride];
        }
        __syncthreads();
    }

    if (localId == 0) {
        int n = count[0];
        if (n > 1) {
            float mean = sumIter[0] / n;
            float meanSq = sumIterSq[0] / n;
            float variance = meanSq - mean * mean;
            float stddev = sqrtf(variance > 0.0f ? variance : 0.0f);
            complexity[tileY * tilesX + tileX] = stddev;
        } else {
            complexity[tileY * tilesX + tileX] = 0.0f;
        }
    }
}

void computeComplexity(
    const int* iterations,
    float* mean,
    float* stddev,
    int width,
    int height,
    int tileSize
) {
    dim3 threads(tileSize, tileSize);
    dim3 blocks((width + tileSize - 1) / tileSize, (height + tileSize - 1) / tileSize);
    size_t sharedMemSize = 2 * tileSize * tileSize * sizeof(float) + tileSize * tileSize * sizeof(int);

    computeComplexityKernel<<<blocks, threads, sharedMemSize>>>(
        iterations, width, height, stddev, tileSize
    );

    cudaDeviceSynchronize();

    // 🐭 Aktuell wird `mean` nicht gefüllt – vorbereitend leer lassen:
    if (mean) cudaMemset(mean, 0, blocks.x * blocks.y * sizeof(float));
}
