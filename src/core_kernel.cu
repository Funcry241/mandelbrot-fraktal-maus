// Datei: src/core_kernel.cu
// 🐭 Maus-Kommentar: Komplexitätsberechnung via Varianz – robuster, strukturbetonter Auto-Zoom

#include <cstdio>
#include "settings.hpp"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include "core_kernel.h"

__device__ float deviceVarianceThreshold = 1e-6f;

extern "C" void setDeviceVarianceThreshold(float threshold) {
    cudaMemcpyToSymbol(deviceVarianceThreshold, &threshold, sizeof(float));
}

__global__ void testKernel(uchar4* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    img[y * w + x] = make_uchar4((x * 255) / w, (y * 255) / h, 128, 255);
}

extern "C" void launch_debugGradient(uchar4* img, int w, int h) {
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    printf("[INFO] DebugGradient Grid (%d, %d)\n", blocks.x, blocks.y);
    testKernel<<<blocks, threads>>>(img, w, h);
    cudaDeviceSynchronize();
}

__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) return make_uchar4(0, 0, 0, 255);

    float log_zn = logf(zx * zx + zy * zy) * 0.5f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float t = (iter + 1.0f - nu) / maxIter;

    float zoomShift = fmodf(logf(zoom + 2.0f) * 0.07f, 1.0f);

    float r = 0.8f + 0.2f * cosf(6.28318f * (t + zoomShift));
    float g = 0.6f + 0.4f * cosf(6.28318f * (t + zoomShift + 0.3f));
    float b = 0.4f + 0.6f * cosf(6.28318f * (t + zoomShift + 0.6f));

    r = powf(r, 1.5f);
    g = powf(g, 1.5f);
    b = powf(b, 1.5f);

    return make_uchar4(fminf(r * 255.0f, 255.0f), fminf(g * 255.0f, 255.0f), fminf(b * 255.0f, 255.0f), 255);
}

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
    dim3 threads(Settings::TILE_W, Settings::TILE_H);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    if (firstLaunch) {
        printf("[INFO] Launch mandelbrotHybrid: Grid (%d, %d)\n", blocks.x, blocks.y);
        firstLaunch = false;
    }

    mandelbrotHybrid<<<blocks, threads>>>(img, iterations, w, h, zoom, offset, maxIter);
    cudaDeviceSynchronize();
}

// 🐭 Komplexitätsbewertung mit Standardabweichung (streuungsbasiert)
__global__ void computeComplexity(const int* iterations, int w, int h, float* complexity) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W;

    int startX = tileX * Settings::TILE_W;
    int startY = tileY * Settings::TILE_H;

    int localX = threadIdx.x;
    int localY = threadIdx.y;
    int x = startX + localX;
    int y = startY + localY;

    int localId = localY * blockDim.x + localX;

    __shared__ float sumIter[Settings::TILE_W * Settings::TILE_H];
    __shared__ float sumIterSq[Settings::TILE_W * Settings::TILE_H];
    __shared__ int count[Settings::TILE_W * Settings::TILE_H];

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

    // Parallel Reduction für Summe und Summe der Quadrate
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
            variance = variance > 0.0f ? variance : 0.0f; // 👈 Sicherheit gegen numerischen Fehler
            float stddev = sqrtf(variance);

            complexity[tileY * tilesX + tileX] = stddev;
        } else {
            complexity[tileY * tilesX + tileX] = 0.0f;
        }
    }
}

