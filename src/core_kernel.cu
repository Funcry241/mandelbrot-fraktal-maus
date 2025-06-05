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

// 🐭 Farbkodierung für Mandelbrot – jetzt mit dynamischem Hue-Shift
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) {
        // Punkte in der Mandelbrot-Menge -> Schwarz
        return make_uchar4(0, 0, 0, 255);
    }

    // Smooth Iteration Count
    float log_zn = logf(zx * zx + zy * zy) / 2.0f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float smoothIter = iter + 1.0f - nu;

    // Normalisieren
    float t = smoothIter / maxIter;
    t = fmodf(t * 3.0f, 1.0f);  // Sanfte Farbringe

    // 🐭 Hue-Shift abhängig vom Zoom
    float hueShift = fmodf(logf(zoom + 1.0f) * 0.1f, 1.0f);

    // Rainbow-Farbverlauf
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

// 🐭 Verfeinerung für interessante Kacheln
__global__ void refineTile(
    uchar4* img,
    int width, int height,
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

    const float escapeRadius2 = 4.0f;
    while (zx * zx + zy * zy < escapeRadius2 && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt;
        ++iter;
    }

    img[y * width + x] = colorMap(iter, maxIter, zx, zy, zoom);  // 🐭 Zoom-Parameter ergänzt
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

    const float escapeRadius2 = 4.0f;

    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            float cx = (static_cast<float>(x) - width * 0.5f) / zoom + offset.x;
            float cy = (static_cast<float>(y) - height * 0.5f) / zoom + offset.y;
            float zx = 0.0f, zy = 0.0f;
            int iter = 0;

            while (zx * zx + zy * zy < escapeRadius2 && iter < maxIter) {
                float xt = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = xt;
                ++iter;
            }

            localSum += iter;
            ++localCnt;

            img[y * width + x] = colorMap(iter, maxIter, zx, zy, zoom);  // 🐭 Zoom-Parameter ergänzt
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

// 🐭 Parallelisierte Complexity-Messung für Auto-Zoom
__global__ void computeComplexity(
    const uchar4* img,
    int width, int height,
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
    __shared__ int sharedCount[Settings::TILE_W * Settings::TILE_H];

    int count = 0;
    if (x < width && y < height) {
        const uchar4& px = img[y * width + x];
        if (px.x < 250u || px.y < 250u || px.z < 250u) {
            count = 1;
        }
    }
    sharedCount[localId] = count;
    __syncthreads();

    // Reduktion im Shared Memory
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedCount[localId] += sharedCount[localId + stride];
        }
        __syncthreads();
    }

    if (localId == 0) {
        complexity[tileY * tilesX + tileX] = static_cast<float>(sharedCount[0]);
    }
}

