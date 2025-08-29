// =============================== core_kernel.cu ===============================
// Minimal: GPU-Entropy & -Contrast + Host-Wrapper computeCudaEntropyContrast.
// Alles Render-/Slice-/Shading-Zeug entfernt.

#include <cuda_runtime.h>
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_host.hpp"

// ----------------- entropy & contrast (coarse metrics) -----------------------
__global__ void entropyKernel(
    const int* it, float* eOut,
    int w, int h, int tile, int maxIter)
{
    int tX = blockIdx.x, tY = blockIdx.y;
    int startX = tX * tile, startY = tY * tile;

    int tilesX = (w + tile - 1) / tile;
    int tileIndex = tY * tilesX + tX;

    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) histo[i] = 0;
    __syncthreads();

    const int total = tile * tile;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int dx = idx % tile, dy = idx / tile;
        int x = startX + dx, y = startY + dy;
        if (x >= w || y >= h) continue;
        int v = it[y * w + x];
        v = max(0, v);
        int bin = min(v * 256 / (maxIter + 1), 255);
        atomicAdd(&histo[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float entropy = 0.0f;
        for (int i = 0; i < 256; ++i) {
            float p = float(histo[i]) / float(total);
            if (p > 0.0f) entropy -= p * __log2f(p);
        }
        eOut[tileIndex] = entropy;
    }
}

__global__ void contrastKernel(
    const float* e, float* cOut,
    int tilesX, int tilesY)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty * tilesX + tx;
    float center = e[idx], sum = 0.0f;
    int cnt = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = tx + dx, ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            int nIdx = ny * tilesX + nx;
            sum += fabsf(e[nIdx] - center);
            ++cnt;
        }

    cOut[idx] = (cnt > 0) ? (sum / cnt) : 0.0f;
}

// ---------------- host wrapper: entropy/contrast -----------------------------
void computeCudaEntropyContrast(
    const int* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter)
{
    const int tilesX = (w + tile - 1) / tile;
    const int tilesY = (h + tile - 1) / tile;
    const int tilesTotal = tilesX * tilesY;

    cudaMemset(d_e, 0, tilesTotal * sizeof(float));

    if (Settings::performanceLogging || Settings::debugLogging) {
        cudaEvent_t evStart, evMid, evEnd;
        cudaEventCreate(&evStart); cudaEventCreate(&evMid); cudaEventCreate(&evEnd);

        cudaEventRecord(evStart, 0);
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        cudaEventRecord(evMid, 0);

        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
        cudaEventRecord(evEnd, 0);
        cudaEventSynchronize(evEnd);

        float ms1=0.f, ms2=0.f;
        cudaEventElapsedTime(&ms1, evStart, evMid);
        cudaEventElapsedTime(&ms2, evMid, evEnd);

        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] en=%.2f ct=%.2f", ms1, ms2);
        } else {
            LUCHS_LOG_HOST("[TIME] en=%.2f | ct=%.2f", ms1, ms2);
        }

        cudaEventDestroy(evStart); cudaEventDestroy(evMid); cudaEventDestroy(evEnd);
    } else {
        entropyKernel<<<dim3(tilesX, tilesY), 128>>>(d_it, d_e, w, h, tile, maxIter);
        contrastKernel<<<dim3((tilesX + 15) / 16, (tilesY + 15) / 16), dim3(16,16)>>>(d_e, d_c, tilesX, tilesY);
    }
}
