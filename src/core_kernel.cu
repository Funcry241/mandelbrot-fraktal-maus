// Datei: src/core_kernel.cu
// Zeilen: 293
// 🐭 Maus-Kommentar: Capybara+Kiwi+MausZoom – Mapping jetzt 100% aspect-korrekt: spanY=spanX*h/w. Keine Verzerrung mehr, Mandelbrot mittig. Otter-Check: Testpixel und Supersampling weiterhin geschützt.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cmath>
#include "common.hpp"
#include "core_kernel.h"
#include "settings.hpp"

// ---- FARB-MAPPING ----
__device__ __forceinline__ uchar4 elegantColor(float t) {
    t = sqrtf(fminf(fmaxf(t, 0.0f), 1.0f));
    float r = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.0f));
    float g = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.33f));
    float b = 0.5f + 0.5f * __sinf(6.2831f * (t + 0.66f));
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

// ---- MANDELBROT-ITERATION ----
__device__ int mandelbrotIterations(float x0, float y0, int maxIter, float& fx, float& fy) {
    float x = 0.0f, y = 0.0f; int i = 0;
    while (x * x + y * y <= 4.0f && i < maxIter) {
        float xt = x * x - y * y + x0;
        y = 2.0f * x * y + y0; x = xt; ++i;
    }
    fx = x; fy = y; return i;
}

// ---- MANDELBROT KERNEL ----
__global__ void mandelbrotKernelAdaptive(
    uchar4* out, int* iterOut, int w, int h, float zoom, float2 offset, int maxIter, int tile, int* super)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * w + x;
    if (x >= w || y >= h || idx < 0 || idx >= w * h) {
        if (iterOut && idx >= 0 && idx < w * h) iterOut[idx] = 0;
        if (out && idx >= 0 && idx < w * h) out[idx] = make_uchar4(0, 0, 0, 255);
        return;
    }
    // Testpixel
    if (x == 0 && y == 0) { iterOut[idx] = 1234; out[idx] = make_uchar4(255, 0, 0, 255);}
    if (x == 1 && y == 0) { iterOut[idx] = 4321; out[idx] = make_uchar4(0, 255, 0, 255);}
    if (x == 2 && y == 0) { iterOut[idx] = 999;  out[idx] = make_uchar4(0, 0, 255, 255);}
    // Mandelbrot-Logik für alle anderen Pixel:
    int tilesX = (w + tile - 1) / tile;
    int tileIndex = (y / tile) * tilesX + (x / tile);
    int S = super ? super[tileIndex] : 1;
    if (S < 1 || S > 32) S = 1;
    float tSum = 0.0f; int iSum = 0;
    float scale = 1.0f / zoom;
    float spanX = 3.5f * scale;
    float spanY = spanX * h / w; // <--- Das fixiert das Aspect Ratio!
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) {
            float dx = (i + 0.5f) / S, dy = (j + 0.5f) / S;
            float fx = ((x + dx) / w - 0.5f) * spanX + offset.x;
            float fy = ((y + dy) / h - 0.5f) * spanY + offset.y;
            float zx, zy; int it = mandelbrotIterations(fx, fy, maxIter, zx, zy); iSum += it;
            float norm = zx * zx + zy * zy;
            float tt = (it + 1.0f - log2f(log2f(fmaxf(norm, 1e-8f)))) / maxIter;
            tSum += fminf(fmaxf(tt, 0.0f), 1.0f);
        }
    out[idx] = elegantColor(tSum / (S * S));
    iterOut[idx] = max(0, iSum / (S * S));
}

// ---- ENTROPY & CONTRAST ----
__global__ void entropyKernel(const int* it, float* eOut, int w, int h, int tile, int maxIter) {
    int tX = blockIdx.x, tY = blockIdx.y, startX = tX * tile, startY = tY * tile;
    __shared__ int histo[256]; for (int i = threadIdx.x; i < 256; i += blockDim.x) histo[i] = 0; __syncthreads();
    int total = tile * tile, tid = threadIdx.x, threads = blockDim.x, local = 0;
    for (int idx = tid; idx < total; idx += threads) {
        int dx = idx % tile, dy = idx / tile, x = startX + dx, y = startY + dy;
        if (x >= w || y >= h) continue; int v = it[y * w + x]; v = max(0, v);
        int bin = min(v * 256 / (maxIter + 1), 255); atomicAdd(&histo[bin], 1); local++;
    } __syncthreads();
    if (threadIdx.x == 0) {
        float entropy = 0.0f; int used = 0;
        for (int i = 0; i < 256; ++i) {
            float p = (local > 0 ? histo[i] / float(total) : 0.0f);
            if (p > 0.0f) entropy -= p * log2f(p);
            used += histo[i];
        }
        int tilesX = (w + tile - 1) / tile, tileIndex = tY * tilesX + tX;
        eOut[tileIndex] = (used > 0) ? entropy : 0.0f;
    }
}

__global__ void contrastKernel(const float* e, float* cOut, int tilesX, int tilesY) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x, ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return; int idx = ty * tilesX + tx; float center = e[idx], sum = 0.0f; int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue; int nx = tx + dx, ny = ty + dy;
        if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue; int nIdx = ny * tilesX + nx; sum += fabsf(e[nIdx] - center); cnt++;
    } cOut[idx] = (cnt > 0) ? sum / cnt : 0.0f;
}

// ---- HOST-WRAPPER: Entropie und Kontrast ----
void computeCudaEntropyContrast(const int* d_it, float* d_e, float* d_c, int w, int h, int tile, int maxIter) {
    int tilesX = (w + tile - 1) / tile, tilesY = (h + tile - 1) / tile;
    entropyKernel <<<dim3(tilesX, tilesY), 128>>> (d_it, d_e, w, h, tile, maxIter); cudaDeviceSynchronize();
    contrastKernel <<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16, 16)>>> (d_e, d_c, tilesX, tilesY); cudaDeviceSynchronize();
}

// ---- HOST-WRAPPER: Mandelbrot+Supersampling ----
void launch_mandelbrotHybrid(uchar4* out, int* d_it, int w, int h, float zoom, float2 offset, int maxIter, int tile, int* d_sup, int supersampling) {
    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    if (Settings::debugLogging) {
        if (d_sup) {
            std::printf("[DEBUG] Mandelbrot-Kernel Call: width=%d, height=%d, maxIter=%d, zoom=%.2f, offset=(%.10f,%.10f), tileSize=%d, supersampling=Buffer, block=(%d,%d), grid=(%d,%d)\n",
                w, h, maxIter, zoom, offset.x, offset.y, tile, block.x, block.y, grid.x, grid.y);
        } else {
            std::printf("[DEBUG] Mandelbrot-Kernel Call: width=%d, height=%d, maxIter=%d, zoom=%.2f, offset=(%.10f,%.10f), tileSize=%d, supersampling=%d, block=(%d,%d), grid=(%d,%d)\n",
                w, h, maxIter, zoom, offset.x, offset.y, tile, supersampling, block.x, block.y, grid.x, grid.y);
        }
    }
    mandelbrotKernelAdaptive<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter, tile, d_sup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    if (Settings::debugLogging) {
        int it[10] = { 0 }; cudaMemcpy(it, d_it, 10 * sizeof(int), cudaMemcpyDeviceToHost); bool inv = false; std::printf("[KERNEL] Iterations First10: ");
        for (int i = 0; i < 10; ++i) { std::printf("%d ", it[i]); if (it[i] < 0) inv = true; }
        if (inv) std::printf("[WARN] Found <0 value! Check buffer init or kernel OOB.\n");
        std::puts("");
    }
}
