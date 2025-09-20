///// Otter: Deterministic CUDA path with single-line ASCII logs per event.
///// Schneefuchs: Device logs via LUCHS_LOG_DEVICE; snprintf only for message construction.
///// Maus: One final macro call; no printf/fprintf in device code.
///// Datei: src/core_kernel.cu

#include "pch.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include "core_kernel.h"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"

// ---------------------- CONST reference-orbit buffer -------------------------
__constant__ double2 zrefConst[Settings::zrefMaxLen];
static_assert(Settings::zrefMaxLen > 0, "Settings::zrefMaxLen must be > 0");

// --------------------------------- helpers -----------------------------------
static __device__ __forceinline__ int clamp_int_0_255(int v) {
    v = (v < 0) ? 0 : v;
    return (v > 255) ? 255 : v;
}

static __device__ __forceinline__ void store_streaming_f32(float* addr, float value) {
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

namespace {
    constexpr int EN_BLOCK_THREADS = 256;
    constexpr int EN_BINS          = 256;
    constexpr int WARP_SIZE        = 32;
    constexpr int EN_WARPS         = EN_BLOCK_THREADS / WARP_SIZE;
    static_assert(EN_WARPS * WARP_SIZE == EN_BLOCK_THREADS, "block size must be multiple of warp size");
}

// ------------------------------- entropy kernel ------------------------------
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

    for (int i = lane; i < EN_BINS; i += WARP_SIZE) {
        histo[warp][i] = 0;
    }
    __syncthreads();

    const float scale = 256.0f / float(maxIter + 1);

    const int totalCells = tile * tile;
    for (int idx = threadIdx.x; idx < totalCells; idx += blockDim.x) {
        const int dx = idx % tile;
        const int dy = idx / tile;
        const int x  = startX + dx;
        const int y  = startY + dy;
        if (x >= w || y >= h) continue;

        int v = (int)__ldg(&it[y * w + x]);
        v = (v < 0) ? 0 : v;
        int bin = __float2int_rz(float(v) * scale);
        bin = clamp_int_0_255(bin);
        atomicAdd(&histo[warp][bin], 1);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < EN_BINS; b += blockDim.x) {
        int sum = 0;
        #pragma unroll
        for (int widx = 0; widx < EN_WARPS; ++widx) sum += histo[widx][b];
        histo[0][b] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
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
        store_streaming_f32(&eOut[tileIndex], entropy);
    }
}

// ------------------------------- contrast kernel -----------------------------
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

    const float val = (cnt > 0) ? (sum / cnt) : 0.0f;
    store_streaming_f32(&cOut[idx], val);
}

// --------------------------- host wrapper: E/C only ---------------------------
#define EC_NT_CHECK(call) \
    do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
        LUCHS_LOG_HOST("[CUDA][EC] rc=%d at %s:%d", (int)_e, __FILE__, __LINE__); } } while (0)

extern "C" void computeCudaEntropyContrast(
    const uint16_t* d_it, float* d_e, float* d_c,
    int w, int h, int tile, int maxIter,
    cudaStream_t stream /*= nullptr*/,
    cudaEvent_t  ecDoneEvent /*= nullptr*/)
{
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

    if (!d_it || !d_e || !d_c) {
        const int    tilesX0     = (w + tile - 1) / tile;
        const int    tilesY0     = (h + tile - 1) / tile;
        const size_t tilesTotal0 = size_t(tilesX0) * size_t(tilesY0);
        if (d_e) EC_NT_CHECK(cudaMemsetAsync(d_e, 0, tilesTotal0 * sizeof(float), stream));
        if (d_c) EC_NT_CHECK(cudaMemsetAsync(d_c, 0, tilesTotal0 * sizeof(float), stream));
        if (ecDoneEvent) EC_NT_CHECK(cudaEventRecord(ecDoneEvent, stream));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EC] null-ptr guard it=%p e=%p c=%p -> zeroed tiles=%zu stream=%p evt=%p",
                           (void*)d_it, (void*)d_e, (void*)d_c, tilesTotal0, (void*)stream, (void*)ecDoneEvent);
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

    EC_NT_CHECK(cudaMemsetAsync(d_e, 0, tilesTotal * sizeof(float), stream));

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

    entropyKernel<<<enGrid, enBlock, 0, stream>>>(d_it, d_e, w, h, tile, maxIter);
    (void)cudaPeekAtLastError();

    contrastKernel<<<ctGrid, ctBlock, 0, stream>>>(d_e, d_c, tilesX, tilesY);
    (void)cudaPeekAtLastError();

    if (ecDoneEvent) {
        EC_NT_CHECK(cudaEventRecord(ecDoneEvent, stream));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EC] done-event recorded stream=%p evt=%p", (void*)stream, (void*)ecDoneEvent);
        }
    }
}
