// 🐭 Maus-Kommentar: CUDA-Kernel für Mandelbrot-Fraktal und Entropieanalyse pro Tile
// - `launch_mandelbrotHybrid`: rendert Fraktalbild + Iterationen
// - `computeTileEntropy`: misst Entropie je Tile zur Bewertung der Bildstruktur (für Auto-Zoom)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "settings.hpp"
#include "core_kernel.h"

// 🧾 Vorwärtsdeklaration notwendig, da Kernel unterhalb verwendet wird
__global__ void mandelbrotKernel(uchar4* output, int* iterationsOut,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIterations);

__device__ int mandelbrotIterations(float x0, float y0, int maxIter) {
    float x = 0.0f, y = 0.0f;
    int iter = 0;
    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        ++iter;
    }
    return iter;
}

// 🚀 Fraktalrendering mit Iterationspuffer (für spätere Analyse)
extern "C" void launch_mandelbrotHybrid(uchar4* output, int* d_iterations,
                                        int width, int height,
                                        float zoom, float2 offset,
                                        int maxIterations) {
    dim3 block(Settings::TILE_W, Settings::TILE_H);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    mandelbrotKernel<<<grid, block>>>(output, d_iterations,
                                      width, height,
                                      zoom, offset,
                                      maxIterations);
    cudaDeviceSynchronize();
}

// 🧠 CUDA-Kernel für Mandelbrot-Iteration pro Pixel
__global__ void mandelbrotKernel(uchar4* output, int* iterationsOut,
                                 int width, int height,
                                 float zoom, float2 offset,
                                 int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float jx = (x - width  / 2.0f) / zoom + offset.x;
    float jy = (y - height / 2.0f) / zoom + offset.y;

    int iter = mandelbrotIterations(jx, jy, maxIterations);
    iterationsOut[y * width + x] = iter;

    float t = iter / (float)maxIterations;
    uchar4 color = make_uchar4(255 * t, 180 * t, 255 * (1.0f - t), 255);
    output[y * width + x] = color;
}

// 📊 Berechnet Entropie pro Tile auf Basis der Iterationsvielfalt
__global__ void entropyKernel(const int* iterations, float* entropyOut,
                              int width, int height, int tileSize,
                              int maxIter) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;

    int startX = tileX * tileSize;
    int startY = tileY * tileSize;

    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        histo[i] = 0;
    __syncthreads();

    int count = 0;

    for (int dy = 0; dy < tileSize; ++dy) {
        for (int dx = 0; dx < tileSize; ++dx) {
            int x = startX + dx;
            int y = startY + dy;
            if (x >= width || y >= height) continue;

            int iter = iterations[y * width + x];
            int bin = min(iter * 256 / (maxIter + 1), 255);
            atomicAdd(&histo[bin], 1);
            ++count;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float entropy = 0.0f;

        if (count > 0) {
            for (int i = 0; i < 256; ++i) {
                float p = histo[i] / (float)count;
                if (p > 0.0f)
                    entropy -= p * log2f(p);
            }
        } else {
            // 🚨 Tile ist komplett leer – damit ignorierbar machen
            entropy = -1.0f;
        }

        int tileIndex = tileY * gridDim.x + tileX;
        entropyOut[tileIndex] = entropy;
    }
}

// 🔧 Host-Funktion zum Starten des Entropie-Kernels
extern "C" void computeTileEntropy(const int* d_iterations,
                                   float* d_entropyOut,
                                   int width, int height,
                                   int tileSize,
                                   int maxIter) {
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    dim3 grid(tilesX, tilesY);
    dim3 block(64);  // ⚠️ Mind. 32 Threads für parallele Histogramm-Init

    entropyKernel<<<grid, block>>>(d_iterations, d_entropyOut,
                                   width, height,
                                   tileSize, maxIter);
    cudaDeviceSynchronize();

#if defined(DEBUG) || Settings::debugLogging
    printf("[DEBUG] computeTileEntropy: gestartet mit %d x %d Tiles (tileSize %d)\n", tilesX, tilesY, tileSize);
#endif
}
