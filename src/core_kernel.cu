// Datei: src/core_kernel.cu
// Maus-Kommentar: Hybrid-Mandelbrot-Kernel: Tile-basiert mit adaptiver Dynamic Parallelism

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>     // uchar4, float2
#include <vector_functions.h> // make_uchar4, make_float2
#include <algorithm>

#define DYNAMIC_THRESHOLD 100.0f  // durchschnittliche Iterationen pro Pixel
#define TILE_W 32
#define TILE_H 32

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

    float sumIter = 0.0f;
    int cntPix = 0;

    // Basis-Durchlauf
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

    // Dynamische Verfeinerung
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
            // ** Kein cudaDeviceSynchronize() hier! **
        }
    }
}

// Host-Exports
extern "C" {

// Launcher für den Hybrid-Kernel
__host__ void launch_mandelbrotHybrid(uchar4* img,
                                      int w, int h,
                                      float zoom, float2 offset,
                                      int maxIter)
{
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((w + TILE_W - 1) / TILE_W,
                 (h + TILE_H - 1) / TILE_H);
    mandelbrotHybrid<<<gridDim, blockDim>>>(img, w, h, zoom, offset, maxIter);
    cudaDeviceSynchronize();
}

// Legacy-Funktionen, falls benötigt
__host__ void mandelbrotPersistent(uchar4* img, int width, int height,
                                   float zoom, float2 center, int maxIter)
{
    dim3 block(16,16), grid((width+15)/16,(height+15)/16);
    mandelbrotHybrid<<<grid, block>>>(img, width, height, zoom, center, maxIter);
    cudaDeviceSynchronize();
}

__host__ void computeComplexity(const uchar4* img, int width, int height, float* out)
{
    int count = 0;
    for (int i = 0; i < width * height; ++i) {
        if (img[i].x || img[i].y || img[i].z) ++count;
    }
    *out = float(count) / (width * height);
}

} // extern "C"
