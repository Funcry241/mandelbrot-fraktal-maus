// Datei: src/core_kernel.cu
#include "core_kernel.h"
#include <cstdio>                  // für printf
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>

#define TILE_W 32
#define TILE_H 32
#define DYNAMIC_THRESHOLD 100.0f

// Farb-Mapping
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter) {
    if (iter == maxIter) return make_uchar4(0, 0, 0, 255);
    float t = float(iter) / maxIter;
    unsigned char r = unsigned char(9*(1-t)*t*t*t*255);
    unsigned char g = unsigned char(15*(1-t)*(1-t)*t*t*255);
    unsigned char b = unsigned char(8.5*(1-t)*(1-t)*(1-t)*t*255);
    return make_uchar4(r, g, b, 255);
}

// Nested Kernel: Verfeinerung einer Kachel mit doppelter Iterationszahl
__global__ void refineTile(uchar4* img, int width, int height,
                           float zoom, float2 offset,
                           int startX, int startY,
                           int tileW, int tileH,
                           int maxIter)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tileW || ty >= tileH) return;
    int x = startX + tx, y = startY + ty;
    if (x >= width || y >= height) return;

    float cx = (x - width * 0.5f) / zoom + offset.x;
    float cy = (y - height * 0.5f) / zoom + offset.y;
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx*zx + zy*zy < 4.0f && iter < maxIter) {
        float xt = zx*zx - zy*zy + cx;
        zy = 2.0f*zx*zy + cy;
        zx = xt;
        ++iter;
    }
    img[y * width + x] = colorMap(iter, maxIter);
}

// Haupt-Kernel: Tile-parallel mit adaptiver Rekursion
__global__ void mandelbrotHybrid(uchar4* img,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIter)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX = min(startX + TILE_W, width);
    int endY = min(startY + TILE_H, height);

    // Lokale Summe und Zählung
    float sumIter = 0.0f;
    int cntPix = 0;

    // Thread-strided Loop für Basiszeichnung
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            float cx = (x - width * 0.5f) / zoom + offset.x;
            float cy = (y - height * 0.5f) / zoom + offset.y;
            float zx = 0.0f, zy = 0.0f;
            int iter = 0;
            while (zx*zx + zy*zy < 4.0f && iter < maxIter) {
                float xt = zx*zx - zy*zy + cx;
                zy = 2.0f*zx*zy + cy;
                zx = xt;
                ++iter;
            }
            sumIter += iter;
            ++cntPix;
            img[y * width + x] = colorMap(iter, maxIter);
        }
    }

    // Nur ein Thread pro Block startet Nested Kernel
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float avgIter = sumIter / cntPix;
        if (avgIter > DYNAMIC_THRESHOLD) {
            int tileW = endX - startX;
            int tileH = endY - startY;
            dim3 bs(min(tileW, TILE_W), min(tileH, TILE_H));
            dim3 gs((tileW + bs.x - 1) / bs.x,
                    (tileH + bs.y - 1) / bs.y);
            refineTile<<<gs, bs>>>(img, width, height,
                                  zoom, offset,
                                  startX, startY,
                                  tileW, tileH,
                                  maxIter * 2);
            cudaGetLastError(); // Fehlercheck
        }
    }
}

extern "C" void launch_mandelbrotHybrid(uchar4* img,
                                        int w, int h,
                                        float zoom, float2 offset,
                                        int maxIter)
{
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim ((w + TILE_W - 1) / TILE_W,
                  (h + TILE_H - 1) / TILE_H);
    mandelbrotHybrid<<<gridDim, blockDim>>>(img, w, h, zoom, offset, maxIter);
    cudaGetLastError(); // Fehlercheck
}

// Neuer Complexity-Kernel mit Atomics, zählt nicht‐schwarze Pixel pro Tile
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int idx   = tileY * gridDim.x + tileX;
    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX   = min(startX + TILE_W, width);
    int endY   = min(startY + TILE_H, height);

    // Unique thread‐index innerhalb der Tile
    int x = startX + threadIdx.x;
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        if (x < endX) {
            uchar4 c = img[y * width + x];
            // schwarz?
            if (!(c.x == 0 && c.y == 0 && c.z == 0)) {
                atomicAdd(&complexity[idx], 1.0f);
            }
        }
    }
    // kein __syncthreads() nötig, jeder Thread addiert nur atomic
}
