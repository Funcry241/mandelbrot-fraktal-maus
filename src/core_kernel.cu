// Datei: src/core_kernel.cu
#include "core_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>     // uchar4, float2
#include <vector_functions.h> // make_uchar4, make_float2

#define DYNAMIC_THRESHOLD 100.0f  // durchschnittliche Iterationen pro Pixel

// ─── Farb-Mapping ────────────────────────────────────────────────────────────
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter) {
    if (iter == maxIter) return make_uchar4(0, 0, 0, 255);
    float t = float(iter) / maxIter;
    unsigned char r = unsigned char(9*(1-t)*t*t*t*255);
    unsigned char g = unsigned char(15*(1-t)*(1-t)*t*t*255);
    unsigned char b = unsigned char(8.5*(1-t)*(1-t)*(1-t)*t*255);
    return make_uchar4(r, g, b, 255);
}

// ─── Nested Kernel: Verfeinerung einer Kachel ────────────────────────────────
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
    img[y * width + x] = colorMap(iter, maxIter);
}

// ─── Haupt-Kernel: Tile-parallel mit adaptiver Rekursion ─────────────────────
__global__ void mandelbrotHybrid(uchar4* img,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIter)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX   = min(startX + TILE_W, width);
    int endY   = min(startY + TILE_H, height);

    // Summen und Zähler
    float sumIter = 0.0f;
    int cntPix    = 0;

    // Thread-gestridete Schleife
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

    // Ein Thread pro Block startet ggf. Verfeinerung
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
        }
    }
}

// ─── Komplexitäts-Kernel: zählt nicht-schwarze Pixel pro Tile ─────────────────
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int idx   = tileY * gridDim.x + tileX;

    int x = tileX * TILE_W + threadIdx.x;
    int y = tileY * TILE_H + threadIdx.y;
    if (x < width && y < height) {
        uchar4 p = img[y * width + x];
        // schwarz = keine Iteration losgelöst
        if (!(p.x == 0 && p.y == 0 && p.z == 0)) {
            atomicAdd(&complexity[idx], 1.0f);
        }
    }
}

// ─── Host-Wrapper zum Starten des Hybrid-Kernels ─────────────────────────────
extern "C" void launch_mandelbrotHybrid(uchar4* img,
                                        int w, int h,
                                        float zoom, float2 offset,
                                        int maxIter)
{
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim ((w + TILE_W - 1) / TILE_W,
                  (h + TILE_H - 1) / TILE_H);
    mandelbrotHybrid<<<gridDim, blockDim>>>(img, w, h, zoom, offset, maxIter);
}
