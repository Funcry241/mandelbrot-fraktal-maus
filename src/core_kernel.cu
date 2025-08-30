// =============================== core_kernel.cu ===============================
// Minimal: GPU-Entropy & -Contrast + Host-Wrapper computeCudaEntropyContrast.
// Alles Render-/Slice-/Shading-Zeug entfernt.

#include <cuda_runtime.h>
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_host.hpp"

// 🦊 Schneefuchs: Launch-bounds für planbare Occupancy bei 128 Threads (Bezug zu Schneefuchs).
__global__ __launch_bounds__(128)
void entropyKernel(
    const int* __restrict__ it, float* __restrict__ eOut,
    int w, int h, int tile, int maxIter)
{
    const int tX = blockIdx.x, tY = blockIdx.y;
    const int startX = tX * tile, startY = tY * tile;

    const int tilesX = (w + tile - 1) / tile;
    const int tileIndex = tY * tilesX + tX;

    __shared__ int histo[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) histo[i] = 0;
    __syncthreads();

    // 🦦 Otter: Vorabfaktor statt Division im Hot-Path (Bezug zu Otter).
    const float scale = 256.0f / float(maxIter + 1);

    const int totalCells = tile * tile;
    for (int idx = threadIdx.x; idx < totalCells; idx += blockDim.x) {
        const int dx = idx % tile;
        const int dy = idx / tile;
        const int x  = startX + dx;
        const int y  = startY + dy;
        if (x >= w || y >= h) continue;

        int v = it[y * w + x];
        v = max(0, v);
        int bin = __float2int_rz(float(v) * scale);
        bin = min(bin, 255);
        atomicAdd(&histo[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // 🦊 Schneefuchs: Zähle echte Samples aus dem Histogramm (keine Kanten-Verzerrung).
        int count = 0;
        for (int i = 0; i < 256; ++i) count += histo[i];

        float entropy = 0.0f;
        if (count > 0) {
            const float invCount = 1.0f / float(count);
            for (int i = 0; i < 256; ++i) {
                const float p = float(histo[i]) * invCount;
                if (p > 0.0f) entropy -= p * __log2f(p);
            }
        }
        eOut[tileIndex] = entropy;
    }
}

__global__ void contrastKernel(
    const float* __restrict__ e, float* __restrict__ cOut,
    int tilesX, int tilesY)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int idx = ty * tilesX + tx;
    const float center = e[idx];
    float sum = 0.0f;
    int cnt = 0;

    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = tx + dx, ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            const int nIdx = ny * tilesX + nx;
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
    if (tilesTotal <= 0) return;

    CUDA_CHECK(cudaMemset(d_e, 0, size_t(tilesTotal) * sizeof(float)));

    constexpr int EN_BLOCK_THREADS = 128;
    const dim3 enGrid(tilesX, tilesY);
    const dim3 enBlock(EN_BLOCK_THREADS);

    const dim3 ctBlock(16, 16);
    const dim3 ctGrid((tilesX + ctBlock.x - 1) / ctBlock.x,
                      (tilesY + ctBlock.y - 1) / ctBlock.y);

    if (Settings::performanceLogging || Settings::debugLogging) {
        cudaEvent_t evStart{}, evMid{}, evEnd{};
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evMid));
        CUDA_CHECK(cudaEventCreate(&evEnd));

        CUDA_CHECK(cudaEventRecord(evStart, 0));
        entropyKernel<<<enGrid, enBlock>>>(d_it, d_e, w, h, tile, maxIter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evMid, 0));

        contrastKernel<<<ctGrid, ctBlock>>>(d_e, d_c, tilesX, tilesY);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evEnd, 0));
        CUDA_CHECK(cudaEventSynchronize(evEnd));

        float ms1 = 0.f, ms2 = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms1, evStart, evMid));
        CUDA_CHECK(cudaEventElapsedTime(&ms2, evMid, evEnd));

        if (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] en=%.2f ct=%.2f", ms1, ms2);
        } else {
            LUCHS_LOG_HOST("[TIME] en=%.2f | ct=%.2f", ms1, ms2);
        }

        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evMid));
        CUDA_CHECK(cudaEventDestroy(evEnd));
    } else {
        entropyKernel<<<enGrid, enBlock>>>(d_it, d_e, w, h, tile, maxIter);
        CUDA_CHECK(cudaGetLastError());
        contrastKernel<<<ctGrid, ctBlock>>>(d_e, d_c, tilesX, tilesY);
        CUDA_CHECK(cudaGetLastError());
    }
}
