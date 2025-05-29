// Datei: src/core_kernel.cu
// Maus-Kommentar: High-Performance Mandelbrot mit Persistent Threads, Tile-Blocking und Dynamic Parallelism inklusive Komplexitäts-Kernel.

#include "core_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>     // für uchar4, float2
#include <vector_functions.h> // für make_uchar4, make_float2

// Globaler Tile-Zähler
__device__ int tileIdxGlobal;

#define DYNAMIC_THRESHOLD 100.0f  // durchschnittliche Iterationen, ab der wir tiefer rechnen

// Nested Kernel: Verfeinerung eines einzelnen Tiles mit doppelter Iterationszahl
__global__ void refineTile(uchar4* img, int width, int height,
                           float zoom, float2 offset,
                           int startX, int startY,
                           int tileW, int tileH,
                           int maxIter)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tileW || ty >= tileH) return;

    int x = startX + tx;
    int y = startY + ty;
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

    uchar4 col = (iter == maxIter)
        ? make_uchar4(0, 0, 0, 255)
        : make_uchar4((iter * 5) % 256,
                      (iter * 7) % 256,
                      (iter * 11) % 256,
                      255);
    img[y * width + x] = col;
}

// Haupt-Kernel mit Persistent Threads und Tile-Dispatch
__global__ void mandelbrotPersistent(uchar4* img, int width, int height,
                                     float zoom, float2 offset,
                                     int maxIter)
{
    while (true) {
        int t = atomicAdd(&tileIdxGlobal, 1);
        int tilesX = (width  + TILE_W - 1) / TILE_W;
        int tilesY = (height + TILE_H - 1) / TILE_H;
        int totalTiles = tilesX * tilesY;
        if (t >= totalTiles) break;

        int tileX = t % tilesX;
        int tileY = t / tilesX;
        int startX = tileX * TILE_W;
        int startY = tileY * TILE_H;
        int endX   = min(startX + TILE_W, width);
        int endY   = min(startY + TILE_H, height);

        float sumIter = 0.0f;
        int   cntPix  = 0;

        for (int py = startY; py < endY; ++py) {
            for (int px = startX; px < endX; ++px) {
                float cx = (px - width * 0.5f) / zoom + offset.x;
                float cy = (py - height * 0.5f) / zoom + offset.y;
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
                uchar4 col = (iter == maxIter)
                    ? make_uchar4(0,0,0,255)
                    : make_uchar4((iter*5)%256,(iter*7)%256,(iter*11)%256,255);
                img[py * width + px] = col;
            }
        }

        float avgIter = sumIter / cntPix;
        if (avgIter > DYNAMIC_THRESHOLD) {
            dim3 bsRef(TILE_W, TILE_H);
            dim3 gsRef((endX - startX + TILE_W - 1) / TILE_W,
                       (endY - startY + TILE_H - 1) / TILE_H);
            refineTile<<<gsRef, bsRef>>>(
                img, width, height, zoom, offset,
                startX, startY,
                endX - startX, endY - startY,
                maxIter * 2
            );
        }
    }
}

// Komplexitäts-Kernel: zählt nicht-schwarze Pixel pro Tile
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int idx   = tileY * ((width + TILE_W -1)/TILE_W) + tileX;

    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX   = min(startX + TILE_W, width);
    int endY   = min(startY + TILE_H, height);

    int count = 0;
    // Thread-strided Loop
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            uchar4 c = img[y * width + x];
            if (c.x || c.y || c.z) ++count;
        }
    }

    // Direkte Addition in globalen Speicher
    atomicAdd(&complexity[idx], count);
}
