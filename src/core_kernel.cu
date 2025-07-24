// Datei: src/core_kernel.cu
// 👝 Maus-Kommentar: Alpha 50 – Verbesserte Robustheit & Eleganz. Neue `pixelToComplex`-Funktion für Klarheit, stabilisierte Farbnormalisierung gegen Artefakte, absichernder Kernel-Start & Entropie-Divisionsschutz. Otter: „Schönheit durch Genauigkeit.“

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

// ---- KOORDINATEN-MAPPING ----
__device__ __forceinline__ float2 pixelToComplex(float px, float py, int w, int h, float spanX, float spanY, float2 offset) {
    return make_float2((px / w - 0.5f) * spanX + offset.x,
                       (py / h - 0.5f) * spanY + offset.y);
}

// ---- MANDELBROT-ITERATION ----
__device__ int mandelbrotIterations(float x0, float y0, int maxIter, float& fx, float& fy) {
    float x = 0.0f, y = 0.0f; int i = 0;
    while (x * x + y * y <= 4.0f && i < maxIter) {
        float xt = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xt;
        ++i;
    }
    fx = x; fy = y;
    return i;
}

// ---- MANDELBROT-KERNEL ----
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

    // Debugpixel (Testanzeige)
    if (x == 0 && y == 0) { iterOut[idx] = 1234; out[idx] = make_uchar4(255, 0, 0, 255); }
    if (x == 1 && y == 0) { iterOut[idx] = 4321; out[idx] = make_uchar4(0, 255, 0, 255); }
    if (x == 2 && y == 0) { iterOut[idx] = 999;  out[idx] = make_uchar4(0, 0, 255, 255); }

    int tilesX = (w + tile - 1) / tile;
    int tileIndex = (y / tile) * tilesX + (x / tile);
    int S = super ? super[tileIndex] : 1;
    if (S < 1 || S > 32) S = 1;

    float tSum = 0.0f; int iSum = 0;
    float scale = 1.0f / zoom;
    float spanX = 3.5f * scale;
    float spanY = spanX * h / w;

    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) {
            float dx = (i + 0.5f) / S, dy = (j + 0.5f) / S;
            float2 c = pixelToComplex(x + dx, y + dy, w, h, spanX, spanY, offset);
            float zx, zy; int it = mandelbrotIterations(c.x, c.y, maxIter, zx, zy); iSum += it;
            float norm = zx * zx + zy * zy;
            float tt = it - log2f(log2f(fmaxf(norm, 1.000001f)));
            tt = fminf(fmaxf(tt / maxIter, 0.0f), 1.0f);
            tSum += tt;
        }

    out[idx] = elegantColor(tSum / (S * S));
    iterOut[idx] = max(0, iSum / (S * S));
}

// ---- ENTROPY & CONTRAST ----
__global__ void entropyKernel(const int* it, float* eOut, int w, int h, int tile, int maxIter) {
    int tX = blockIdx.x, tY = blockIdx.y, startX = tX * tile, startY = tY * tile;
    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) histo[i] = 0;
    __syncthreads();

    int total = tile * tile, tid = threadIdx.x, threads = blockDim.x;
    for (int idx = tid; idx < total; idx += threads) {
        int dx = idx % tile, dy = idx / tile, x = startX + dx, y = startY + dy;
        if (x >= w || y >= h) continue;
        int v = it[y * w + x]; v = max(0, v);
        int bin = min(v * 256 / (maxIter + 1), 255);
        atomicAdd(&histo[bin], 1);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        float entropy = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float p = float(histo[i]) / float(total);
            if (p > 0.0f) entropy -= p * log2f(p);
        }
        int tilesX = (w + tile - 1) / tile;
        int tileIndex = tY * tilesX + tX;
        eOut[tileIndex] = entropy;
    }
}

__global__ void contrastKernel(const float* e, float* cOut, int tilesX, int tilesY) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty * tilesX + tx;
    float center = e[idx], sum = 0.0f; int cnt = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = tx + dx, ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            int nIdx = ny * tilesX + nx;
            sum += fabsf(e[nIdx] - center); cnt++;
        }

    cOut[idx] = (cnt > 0) ? sum / cnt : 0.0f;
}

// ---- HOST-WRAPPER: Entropie & Kontrast ----
void computeCudaEntropyContrast(const int* d_it, float* d_e, float* d_c, int w, int h, int tile, int maxIter) {
    int tilesX = (w + tile - 1) / tile, tilesY = (h + tile - 1) / tile;
    entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
    cudaDeviceSynchronize();
    contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16, 16)>>>(d_e, d_c, tilesX, tilesY);
    cudaDeviceSynchronize();
}

// ---- HOST-WRAPPER: Mandelbrot-Hybrid ----
void launch_mandelbrotHybrid(uchar4* out, int* d_it, int w, int h, float zoom, float2 offset, int maxIter, int tile, int* d_sup, int supersampling) {
    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    if (Settings::debugLogging) {
        std::printf("[Kernel] %dx%d | Zoom: %.3e | Offset: (%.5f, %.5f) | Iter: %d | Tile: %d | Supersampling: %s\n",
            w, h, zoom, offset.x, offset.y, maxIter, tile,
            d_sup ? "Buffer" : std::to_string(supersampling).c_str());
    }

    if (out && d_it)
        mandelbrotKernelAdaptive<<<grid, block>>>(out, d_it, w, h, zoom, offset, maxIter, tile, d_sup);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    if (Settings::debugLogging) {
        int it[10] = { 0 };
        cudaMemcpy(it, d_it, sizeof(it), cudaMemcpyDeviceToHost);
        std::printf("[Iter] First10: ");
        for (int i = 0; i < 10; ++i) std::printf("%d ", it[i]);
        std::puts("");
    }
}
