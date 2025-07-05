// Datei: src/core_kernel.cu
// Zeilen: 358
// 🐭 Maus-Kommentar: Projekt Capybara Phase 2 + Kiwi: Konsistente Iterationsspeicherung, Heatmap-Berechnung nach aktuellem Frame. Fix für Thread-0-Kachelproblem am Rand. Otter: „Kiwi bringt Klarheit, Capybara hält die Linie.“
// Maus-Logik: Nach jedem Mandelbrot-Kernel werden die ersten 10 Iterationswerte geloggt, um Kernel-Fehler sichtbar zu machen.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"

__device__ __forceinline__ uchar4 elegantColor(float t) {
    if (t < 0.0f) return make_uchar4(0, 0, 0, 255);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    float intensity = sqrtf(t);
    float r = 0.5f + 0.5f * __sinf(6.2831f * (intensity + 0.0f));
    float g = 0.5f + 0.5f * __sinf(6.2831f * (intensity + 0.33f));
    float b = 0.5f + 0.5f * __sinf(6.2831f * (intensity + 0.66f));
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__device__ int mandelbrotIterations(float x0, float y0, int maxIter, float& finalX, float& finalY) {
    float x = 0.0f, y = 0.0f;
    int iter = 0;
    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        ++iter;
    }
    finalX = x;
    finalY = y;
    return iter;
}

__global__ void mandelbrotKernelAdaptive(uchar4* output, int* iterationsOut,
                                         int width, int height,
                                         float zoom, float2 offset,
                                         int maxIterations,
                                         int tileSize,
                                         const int* tileSupersampling) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int tileX = x / tileSize;
    int tileY = y / tileSize;
    int tilesX = (width + tileSize - 1) / tileSize;
    int tileIndex = tileY * tilesX + tileX;

    int S = tileSupersampling[tileIndex];
    float totalT = 0.0f;
    int totalIter = 0;

    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            float dx = (i + 0.5f) / S;
            float dy = (j + 0.5f) / S;
            float jx = (x + dx - width * 0.5f) / zoom + offset.x;
            float jy = (y + dy - height * 0.5f) / zoom + offset.y;

            float zx, zy;
            int iter = mandelbrotIterations(jx, jy, maxIterations, zx, zy);
            totalIter += iter;

            float norm = zx * zx + zy * zy;
            float t = (iter + 1.0f - log2f(log2f(fmaxf(norm, 1e-8f)))) / maxIterations;
            t = fminf(fmaxf(t, 0.0f), 1.0f);
            totalT += t;
        }
    }

    float avgT = totalT / (S * S);
    int avgIter = totalIter / (S * S);

    output[y * width + x] = elegantColor(avgT);
    // Capybara: Schreibe echte Iterationswerte zurück
    iterationsOut[y * width + x] = avgIter;
}

// Fix: Setze Entropie für *jede* Kachel, auch wenn Thread 0 keine Pixel verarbeitet (Kiwi)
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

    int total = tileSize * tileSize;
    int threads = blockDim.x;
    int tid = threadIdx.x;
    int localCount = 0;

    for (int idx = tid; idx < total; idx += threads) {
        int dx = idx % tileSize;
        int dy = idx / tileSize;
        int x = startX + dx;
        int y = startY + dy;
        if (x >= width || y >= height) continue;
        int iter = iterations[y * width + x];
        int bin = min(iter * 256 / (maxIter + 1), 255);
        atomicAdd(&histo[bin], 1);
        localCount++;
    }
    __syncthreads();

    // Kiwi: Setze Entropie immer, auch wenn localCount == 0 für thread 0, damit Werte definiert bleiben!
    if (threadIdx.x == 0) {
        float entropy = 0.0f;
        int usedCount = 0;
        for (int i = 0; i < 256; ++i) {
            float p = (localCount > 0 ? histo[i] / (float)total : 0.0f);
            if (p > 0.0f)
                entropy -= p * log2f(p);
            usedCount += histo[i];
        }
        int tilesX = (width + tileSize - 1) / tileSize;
        int tileIndex = tileY * tilesX + tileX;
        // Setze Entropie auf 0 falls keine Pixel im Tile waren
        entropyOut[tileIndex] = (usedCount > 0) ? entropy : 0.0f;
    }
}

__global__ void contrastKernel(const float* entropy, float* contrastOut,
                               int tilesX, int tilesY) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty * tilesX + tx;
    float center = entropy[idx];
    float sumDiff = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = tx + dx;
            int ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            int nIdx = ny * tilesX + nx;
            sumDiff += fabsf(entropy[nIdx] - center);
            count++;
        }
    }
    contrastOut[idx] = (count > 0) ? sumDiff / count : 0.0f;
}

extern "C" void computeCudaEntropyContrast(
    const int* d_iterations,
    float* d_entropyOut,
    float* d_contrastOut,
    int width,
    int height,
    int tileSize,
    int maxIter
) {
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    dim3 gridE(tilesX, tilesY);
    dim3 blockE(128);
    entropyKernel<<<gridE, blockE>>>(d_iterations, d_entropyOut, width, height, tileSize, maxIter);
    cudaDeviceSynchronize();

    dim3 gridC((tilesX + 15) / 16, (tilesY + 15) / 16);
    dim3 blockC(16, 16);
    contrastKernel<<<gridC, blockC>>>(d_entropyOut, d_contrastOut, tilesX, tilesY);
    cudaDeviceSynchronize();
}

extern "C" void launch_mandelbrotHybrid(
    uchar4* output,
    int* d_iterations,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    int tileSize,
    const int* d_tileSupersampling
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    mandelbrotKernelAdaptive<<<grid, block>>>(output, d_iterations,
                                              width, height,
                                              zoom, offset,
                                              maxIterations,
                                              tileSize,
                                              d_tileSupersampling);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // --- DEBUG: Iterations-Buffer nach Kernel prüfen (Settings::debugLogging)
    if (Settings::debugLogging) {
        int iters_dbg[10] = {0};
        cudaMemcpy(iters_dbg, d_iterations, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        std::printf("[KERNEL] Iterations First10: ");
        for (int i = 0; i < 10; ++i) std::printf("%d ", iters_dbg[i]);
        std::puts("");
    }
}
