///// Otter: Deterministic CUDA path with single-line ASCII logs per event.
///// Schneefuchs: Device logs via LUCHS_LOG_DEVICE; snprintf only for message construction.
///// Maus: One final macro call; no printf/fprintf in device code.
///// Datei: src/core_kernel.cu

#include "pch.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"

// --------------------------------- helpers -----------------------------------
static __device__ __forceinline__ int clamp_int_0_255(int v) {
    v = (v < 0) ? 0 : v;
    return (v > 255) ? 255 : v;
}

namespace {
    // Keep block size in one place so __launch_bounds__ and host launch stay in sync.
    constexpr int EN_BLOCK_THREADS = 256;   // 256 == EN_BINS -> simple, full parallelism
    constexpr int EN_BINS          = 256;
    constexpr int WARP_SIZE        = 32;
    constexpr int EN_WARPS         = EN_BLOCK_THREADS / WARP_SIZE;
    static_assert(EN_WARPS * WARP_SIZE == EN_BLOCK_THREADS, "block size must be multiple of warp size");
}

// ------------------------------- entropy kernel ------------------------------
// Warp-private histograms to reduce atomic contention (EN_WARPS × 256 bins in shared mem).
__global__ __launch_bounds__(EN_BLOCK_THREADS, 2)
void entropyKernel(
    const uint16_t* __restrict__ it,
    float* __restrict__ eOut,
    int w, int h, int tile, int maxIter)
{
    const int tX = blockIdx.x;
    const int tY = blockIdx.y;

    const int tilesX = (w + tile - 1) / tile;
    const int tilesY = (h + tile - 1) / tile;
    if (tX >= tilesX || tY >= tilesY) return;

    const int startX = tX * tile;
    const int startY = tY * tile;
    const int tileIndex = tY * tilesX + tX;

    __shared__ int histo[EN_WARPS][EN_BINS];

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x >> 5;

    // Zero warp-local histograms
    for (int i = lane; i < EN_BINS; i += WARP_SIZE) {
        histo[warp][i] = 0;
    }
    __syncthreads();

    // Precomputed scale avoids division in the hot path.
    const float scale = 256.0f / float(maxIter + 1);

    const int totalCells = tile * tile;
    for (int idx = threadIdx.x; idx < totalCells; idx += blockDim.x) {
        const int dx = idx % tile;
        const int dy = idx / tile;
        const int x  = startX + dx;
        const int y  = startY + dy;
        if (x >= w || y >= h) continue;

        // read-only cached fetch
        int v = (int)__ldg(&it[y * w + x]);
        v = (v < 0) ? 0 : v;
        int bin = __float2int_rz(float(v) * scale);
        bin = clamp_int_0_255(bin);
        atomicAdd(&histo[warp][bin], 1);
    }
    __syncthreads();

    // Reduce warp-local histograms into histo[0][*]  (STRIDED -> covers all 256 bins)
    for (int b = threadIdx.x; b < EN_BINS; b += blockDim.x) {
        int sum = 0;
        #pragma unroll
        for (int widx = 0; widx < EN_WARPS; ++widx) sum += histo[widx][b];
        histo[0][b] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Exact sample count from the merged histogram
        int count = 0;
        #pragma unroll
        for (int i = 0; i < EN_BINS; ++i) count += histo[0][i];

        float entropy = 0.0f;
        if (count > 0) {
            const float invCount = 1.0f / float(count);
            #pragma unroll
            for (int i = 0; i < EN_BINS; ++i) {
                const float p = float(histo[0][i]) * invCount;
                if (p > 0.0f) entropy -= p * __log2f(p);
            }
        }
        eOut[tileIndex] = entropy;
    }
}

// ------------------------------- contrast kernel -----------------------------
// Launch with 16x16 (256 thr) blocks; predictable occupancy.
__global__ __launch_bounds__(256, 2)
void contrastKernel(
    const float* __restrict__ e,
    float* __restrict__ cOut,
    int tilesX, int tilesY)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int idx = ty * tilesX + tx;
    const float center = e[idx];
    float sum = 0.0f;
    int cnt = 0;

    // 8-neighborhood (without center)
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = tx + dx;
            const int ny = ty + dy;
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            const int nIdx = ny * tilesX + nx;
            sum += fabsf(e[nIdx] - center);
            ++cnt;
        }
    }

    cOut[idx] = (cnt > 0) ? (sum / cnt) : 0.0f;
}

// --------------------------- host wrapper: E/C only ---------------------------
// NOTE (Otter): Fully asynchronous: no host-side synchronization, no device-wide sync.
// - Kernels launch on the provided `stream`.
// - Entropy buffer is cleared with cudaMemsetAsync on the same `stream`.
// - If `ecDoneEvent` is non-null, it is recorded on `stream` after the contrast kernel.

// Non-throwing, numeric-rc logging for extern "C" wrapper (avoid C4297 under /WX)
#define EC_NT_CHECK(call) \
    do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
        LUCHS_LOG_HOST("[CUDA][EC] rc=%d at %s:%d", (int)_e, __FILE__, __LINE__); } } while (0)

extern "C" void computeCudaEntropyContrast(
    const uint16_t* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter,
    cudaStream_t stream /*= nullptr*/,
    cudaEvent_t  ecDoneEvent /*= nullptr*/)
{
    // Early guards: robust zeroing for invalid sizes (async, non-throwing).
    if (w <= 0 || h <= 0 || tile <= 0 || maxIter < 0) {
        const int    tilesX0     = (tile > 0) ? (w + tile - 1) / tile : 0;
        const int    tilesY0     = (tile > 0) ? (h + tile - 1) / tile : 0;
        const size_t tilesTotal0 = size_t(tilesX0) * size_t(tilesY0);
        if (d_e && tilesTotal0) EC_NT_CHECK(cudaMemsetAsync(d_e, 0, tilesTotal0 * sizeof(float), stream));
        if (d_c && tilesTotal0) EC_NT_CHECK(cudaMemsetAsync(d_c, 0, tilesTotal0 * sizeof(float), stream));
        if (ecDoneEvent)        EC_NT_CHECK(cudaEventRecord(ecDoneEvent, stream));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EC] invalid-dims queued zero-fill tiles=%zu stream=%p evt=%p",
                           tilesTotal0, (void*)stream, (void*)ecDoneEvent);
        }
        return;
    }

    const int    tilesX     = (w + tile - 1) / tile;
    const int    tilesY     = (h + tile - 1) / tile;
    const size_t tilesTotal = size_t(tilesX) * size_t(tilesY);
    if (tilesTotal == 0) {
        if (ecDoneEvent) EC_NT_CHECK(cudaEventRecord(ecDoneEvent, stream));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EC] zero-tiles queued (nothing to do) stream=%p evt=%p",
                           (void*)stream, (void*)ecDoneEvent);
        }
        return;
    }

    // Clear entropy buffer (contrast reads neighbors; entropy kernel overwrites all valid tiles).
    EC_NT_CHECK(cudaMemsetAsync(d_e, 0, tilesTotal * sizeof(float), stream));

    // Launch config
    const dim3 enGrid(tilesX, tilesY);
    const dim3 enBlock(EN_BLOCK_THREADS);

    const dim3 ctBlock(16, 16);
    const dim3 ctGrid(
        (tilesX + ctBlock.x - 1) / ctBlock.x,
        (tilesY + ctBlock.y - 1) / ctBlock.y
    );

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        LUCHS_LOG_HOST("[EC] queued tiles=%dx%d (%zu) tileSize=%d maxIter=%d stream=%p",
                       tilesX, tilesY, tilesTotal, tile, maxIter, (void*)stream);
    }

    // Launch kernels on the provided stream (no host sync)
    entropyKernel<<<enGrid, enBlock, 0, stream>>>(d_it, d_e, w, h, tile, maxIter);
    (void)cudaPeekAtLastError();

    contrastKernel<<<ctGrid, ctBlock, 0, stream>>>(d_e, d_c, tilesX, tilesY);
    (void)cudaPeekAtLastError();

    // Record completion event for downstream streams (copy/upload), if requested.
    if (ecDoneEvent) {
        EC_NT_CHECK(cudaEventRecord(ecDoneEvent, stream));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EC] done-event recorded stream=%p evt=%p", (void*)stream, (void*)ecDoneEvent);
        }
    }
}
