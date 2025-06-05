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
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter) {
    float t = static_cast<float>(iter) / maxIter;
    unsigned char r = static_cast<unsigned char>(9 * (1 - t) * t * t * t * 255);
    unsigned char g = static_cast<unsigned char>(15 * (1 - t) * (1 - t) * t * t * 255);
    unsigned char b = static_cast<unsigned char>(8.5f * (1 - t) * (1 - t) * (1 - t) * t * 255);
    return make_uchar4(r, g, b, 255);
}

// 🐭 Verfeinerung für interessante Kacheln
__global__ void refineTile(
    uchar4* img, int width, int height,
    float zoom, float2 offset,
    int startX, int startY,
    int tileW, int tileH,
    int maxIter
) {
    int x = startX + blockIdx.x * blockDim.x + threadIdx.x;
    int y = startY + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (startX + tileW) || y >= (startY + tileH) || x >= width || y >= height) return;

    float cx = (static_cast<float>(x) - width * 0.5f) / zoom + offset.x;
    float cy = (static_cast<float>(y) - height * 0.5f) / zoom + offset.y;
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt;
        ++iter;
    }
    img[y * width + x] = colorMap(iter, maxIter);
}

// 🐭 Mandelbrot Haupt-Kernel
__global__ void mandelbrotHybrid(
    uchar4* img,
    int width, int height,
    float zoom, float2 offset,
    int maxIter
) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int startX = tileX * Settings::TILE_W;
    int startY = tileY * Settings::TILE_H;
    int endX = min(startX + Settings::TILE_W, width);
    int endY = min(startY + Settings::TILE_H, height);

    float localSum = 0.0f;
    int   localCnt = 0;

    __shared__ float blockSum;
    __shared__ int   blockCnt;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        blockSum = 0.0f;
        blockCnt = 0;
    }
    __syncthreads();

    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            float cx = (static_cast<float>(x) - width * 0.5f) / zoom + offset.x;
            float cy = (static_cast<float>(y) - height * 0.5f) / zoom + offset.y;
            float zx = 0.0f, zy = 0.0f;
            int iter = 0;
            while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
                float xt = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = xt;
                ++iter;
            }
            localSum += iter;
            ++localCnt;
            img[y * width + x] = colorMap(iter, maxIter);
        }
    }

    atomicAdd(&blockSum, localSum);
    atomicAdd(&blockCnt, localCnt);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float avgIter = blockSum / static_cast<float>(blockCnt);
        if (avgIter > Settings::DYNAMIC_THRESHOLD) {
            int tileW = endX - startX;
            int tileH = endY - startY;
            dim3 bs(min(tileW, Settings::TILE_W), min(tileH, Settings::TILE_H));
            dim3 gs((tileW + bs.x - 1) / bs.x,
                    (tileH + bs.y - 1) / bs.y);
            refineTile<<<gs, bs>>>(
                img, width, height,
                zoom, offset,
                startX, startY,
                tileW, tileH,
                maxIter * 2
            );
            cudaError_t errNested = cudaGetLastError();
            if (errNested != cudaSuccess) {
                printf("[NESTED ERROR] refineTile: %s\n", cudaGetErrorString(errNested));
            } else {
                printf("[INFO] refineTile gestartet (TileX: %d, TileY: %d)\n", tileX, tileY);
            }
        }
    }
}

// 🐭 Mandelbrot-Launcher
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,
    int width, int height,
    float zoom, float2 offset,
    int maxIter
) {
    int tilesX = (width  + Settings::TILE_W - 1) / Settings::TILE_W;
    int tilesY = (height + Settings::TILE_H - 1) / Settings::TILE_H;
    dim3 blocks(tilesX, tilesY);
    dim3 threads(Settings::TILE_W, Settings::TILE_H);

    printf("[INFO] Starte mandelbrotHybrid mit Grid (%d,%d) und Threads (%d,%d)\n", blocks.x, blocks.y, threads.x, threads.y);

    mandelbrotHybrid<<<blocks, threads>>>(img, width, height, zoom, offset, maxIter);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[LAUNCH ERROR] mandelbrotHybrid: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// 🐭 Complexity-Messung für Auto-Zoom
__global__ void computeComplexity(
    const uchar4* img,
    int width, int height,
    float* complexity
) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tilesX = (width  + Settings::TILE_W - 1) / Settings::TILE_W;

    int startX = tileX * Settings::TILE_W;
    int startY = tileY * Settings::TILE_H;
    int endX = min(startX + Settings::TILE_W, width);
    int endY = min(startY + Settings::TILE_H, height);

    float count = 0.0f;
    for (int y = startY; y < endY; ++y) {
        int baseIdx = y * width;
        for (int x = startX; x < endX; ++x) {
            const uchar4& px = img[baseIdx + x];
            if (px.w != 0u) {
                count += 1.0f;
            }
        }
    }
    complexity[tileY * tilesX + tileX] = count;
}
