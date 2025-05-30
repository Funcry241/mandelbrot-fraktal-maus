// Datei: src/core_kernel.cu

#include "core_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>              // für printf in Device‐Code

// Farb‐Mapping bleibt unverändert…
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter) {
    if (iter == maxIter) return make_uchar4(0,0,0,255);
    float t = float(iter)/maxIter;
    unsigned char r = unsigned char(9*(1-t)*t*t*t*255);
    unsigned char g = unsigned char(15*(1-t)*(1-t)*t*t*255);
    unsigned char b = unsigned char(8.5*(1-t)*(1-t)*(1-t)*t*255);
    return make_uchar4(r,g,b,255);
}

// …und der Hybrid‐Kernel bleibt so, wie Du ihn hast.
__global__ void mandelbrotHybrid(uchar4* img,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIter)
{
    // … Dein bestehender Code …
}

// Launch‐Wrapper
extern "C" void launch_mandelbrotHybrid(uchar4* img,
                                        int w, int h,
                                        float zoom, float2 offset,
                                        int maxIter)
{
    dim3 bd(TILE_W, TILE_H);
    dim3 gd((w + TILE_W-1)/TILE_W, (h + TILE_H-1)/TILE_H);
    mandelbrotHybrid<<<gd,bd>>>(img,w,h,zoom,offset,maxIter);
    cudaGetLastError();
}

// **Neu: Komplexitäts‐Kernel**
// Zählt pro Tile alle Nicht‐Schwarz‐Pixel (x|y|z != 0) mit atomicAdd
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX   = min(startX + TILE_W, width);
    int endY   = min(startY + TILE_H, height);

    // Thread‐lokale Zählung
    float localSum = 0.0f;
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            uchar4 p = img[y * width + x];
            if (p.x | p.y | p.z) localSum += 1.0f;
        }
    }

    // Alle Threads addieren ihr localSum in complexity[tileIndex]
    __shared__ float blockSum[TILE_W*TILE_H];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    blockSum[tid] = localSum;
    __syncthreads();

    // Parallel‐Reduction im Block
    for (int s = (blockDim.x*blockDim.y)/2; s > 0; s >>= 1) {
        if (tid < s) {
            blockSum[tid] += blockSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int tileIdx = tileY * gridDim.x + tileX;
        atomicAdd(&complexity[tileIdx], blockSum[0]);
    }
    // Fehlercheck
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess && tid==0) {
        printf("computeComplexity launch failed: %s\n", cudaGetErrorString(err));
    }
}
