// Datei: src/core_kernel.cu
// Zeilen: 394
// 🐭 Maus-Kommentar: Capybara+Kiwi+MausZoom – Device-Debug für Fraktal-Koord, robust gegen OOB, Kernels immer CUDA-Error-checked. Otter liebt ASCII-Logs und float-Klarheit!

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
                                         int* tileSupersampling) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Debug-Injection: Schreibe Testwert und Device-Debug für Pixel 0,0 & 1,0/0,1/1,1
#if 1
    if (x < 2 && y < 2) {
        float dx = 0.5f, dy = 0.5f;
        float jx = (x + dx - width * 0.5f) / zoom + offset.x;
        float jy = (y + dy - height * 0.5f) / zoom + offset.y;
        float zx, zy;
        int iter = mandelbrotIterations(jx, jy, maxIterations, zx, zy);
        printf("[DEVICE] Pixel(%d,%d): jx=%.8f jy=%.8f iter=%d\n", x, y, jx, jy, iter);

        if (x == 0 && y == 0) {
            iterationsOut[idx] = 1234; // Testwert, sollte im [KERNEL] Iterations First10: sichtbar sein!
            output[idx] = make_uchar4(255, 0, 0, 255); // Rot für links oben
            // Kein return, restlicher Mandelbrot-Code läuft normal weiter
        }
    }
#endif

    // OOB-Guard: Immer gültigen Wert schreiben
    if (x >= width || y >= height) {
        if (idx < width * height && iterationsOut) iterationsOut[idx] = 0;
        if (output && idx < width * height) output[idx] = make_uchar4(0, 0, 0, 255);
        return;
    }

    int tileX = x / tileSize;
    int tileY = y / tileSize;
    int tilesX = (width + tileSize - 1) / tileSize;
    int tileIndex = tileY * tilesX + tileX;
    int S = (tileSupersampling ? tileSupersampling[tileIndex] : 1);
    float totalT = 0.0f;
    int totalIter = 0;

    for (int i = 0; i < S; ++i) {
        float dx = (i + 0.5f) / S;
        for (int j = 0; j < S; ++j) {
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

    float invS2 = 1.0f / (S * S);
    output[idx] = elegantColor(totalT * invS2);
    iterationsOut[idx] = max(0, (int)(totalIter * invS2));
}

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
        iter = max(0, iter);
        int bin = min(iter * 256 / (maxIter + 1), 255);
        atomicAdd(&histo[bin], 1);
        localCount++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float entropy = 0.0f;
        int usedCount = 0;
        for (int i = 0; i < 256; ++i) {
            float p = (localCount > 0 ? histo[i] / (float)total : 0.0f);
            if (p > 0.0f) entropy -= p * log2f(p);
            usedCount += histo[i];
        }
        int tilesX = (width + tileSize - 1) / tileSize;
        int tileIndex = tileY * tilesX + tileX;
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

void computeCudaEntropyContrast(
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
    cudaError_t errE = cudaGetLastError();
    if (errE != cudaSuccess) {
        printf("[CUDA ERROR] entropyKernel: %s\n", cudaGetErrorString(errE));
    }
    cudaDeviceSynchronize();

    dim3 gridC((tilesX + 15) / 16, (tilesY + 15) / 16);
    dim3 blockC(16, 16);
    contrastKernel<<<gridC, blockC>>>(d_entropyOut, d_contrastOut, tilesX, tilesY);
    cudaError_t errC = cudaGetLastError();
    if (errC != cudaSuccess) {
        printf("[CUDA ERROR] contrastKernel: %s\n", cudaGetErrorString(errC));
    }
    cudaDeviceSynchronize();
}

void launch_mandelbrotHybrid(
    uchar4* output,
    int* d_iterations,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    int tileSize,
    int* d_tileSupersampling,
    int supersampling
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Mandelbrot-Kernel Call: width=%d, height=%d, maxIter=%d, zoom=%.2f, offset=(%.10f, %.10f), tileSize=%d, supersampling=%d, block=(%d,%d), grid=(%d,%d)\n",
            width, height, maxIterations, zoom, offset.x, offset.y, tileSize,
            (d_tileSupersampling ? -42 : 1),
            block.x, block.y, grid.x, grid.y
        );
    }

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

    if (Settings::debugLogging) {
        int iters_dbg[10] = {0};
        cudaMemcpy(iters_dbg, d_iterations, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        bool anyInvalid = false;
        for (int i = 0; i < 10; ++i) if (iters_dbg[i] < 0) anyInvalid = true;
        std::printf("[KERNEL] Iterations First10: ");
        for (int i = 0; i < 10; ++i) std::printf("%d ", iters_dbg[i]);
        if (anyInvalid)
            std::printf("[WARN] Found <0 value! Check buffer init or kernel OOB.\n");
        std::puts("");
    }
}
