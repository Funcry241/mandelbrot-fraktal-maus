///// Otter: Nacktmull – GPU heatmap metrics (boundary + contrast), deterministic single-thread-per-tile + pinned host slab
///// Schneefuchs: One kernel path; 1×1 block per tile (stable FP sums); numeric rc logs; no GL; ASCII-only; reuse buffers
///// Maus: Identical results; no fast-math; device slab + pinned host slab; immediate sync then memcpy to vectors
///// Datei: src/heatmap_metrics.cu

#include "pch.hpp"
#include "heatmap_metrics.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>   // std::memcpy
#include <cstdint>   // uint16_t

// integer helpers (avoid <algorithm> overload ambiguity in device code)
static __device__ __forceinline__ int i_min(int a, int b) { return a < b ? a : b; }
static __device__ __forceinline__ int i_max(int a, int b) { return a > b ? a : b; }

// -------------------------------- kernel --------------------------------
// Channel 0: boundary score = pEsc * (1 - pEsc)
// Channel 1: contrast      = stddev(iterations)
// Nacktmull note: We keep a single thread per tile to preserve deterministic
// FP summation order across runs/architectures (no parallel reduction noise).
__global__ void kernel_tile_metrics(const uint16_t* __restrict__ it,
                                    int w, int h,
                                    int tilePx, int tilesX, int tilesY,
                                    float* __restrict__ boundaryOut,
                                    float* __restrict__ contrastOut)
{
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int x0 = tx * tilePx;
    const int y0 = ty * tilePx;
    const int x1 = i_min(w, x0 + tilePx);
    const int y1 = i_min(h, y0 + tilePx);

    const int tileW = i_max(0, x1 - x0);
    const int tileH = i_max(0, y1 - y0);
    const int nPix  = tileW * tileH;
    const int outIx = ty * tilesX + tx;

    if (nPix <= 0) {
        if (boundaryOut) boundaryOut[outIx] = 0.0f;
        if (contrastOut) contrastOut[outIx] = 0.0f;
        return;
    }

    // Ein Pass: Summe, Summe^2
    double sum = 0.0;
    double sum2 = 0.0;

    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            const double v = (double)row[x];
            sum  += v;
            sum2 += v * v;
        }
    }
    const double invN = 1.0 / (double)nPix;
    const double mean = sum * invN;

    // Escape-Anteil grob per Mittelwertschwelle
    int countEsc = 0;
    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            if ((double)row[x] < mean) countEsc++;
        }
    }

    // Kontrast = Standardabweichung
    double var = sum2 * invN - mean * mean;
    if (var < 0.0) var = 0.0;

    // Boundary-Score
    const float pEsc = (float)countEsc * (float)invN;
    const float boundary = pEsc * (1.0f - pEsc);

    if (boundaryOut) boundaryOut[outIx] = boundary;
    if (contrastOut) contrastOut[outIx] = (float)sqrt(var);
}

// --------------- device slab buffer for boundary+contrast ----------------
static float* s_dMetrics = nullptr;   // layout: [tiles] boundary | [tiles] contrast
static size_t s_tilesCap = 0;

// --------------- pinned host slab for boundary+contrast ------------------
static float* s_hPinned = nullptr;    // layout: [tiles] boundary | [tiles] contrast
static size_t s_hostCap = 0;

static bool ensureDeviceBuffers(size_t tiles) {
    if (tiles <= s_tilesCap && s_dMetrics) return true;
    if (s_dMetrics) { cudaFree(s_dMetrics); s_dMetrics = nullptr; }
    s_tilesCap = 0;

    const size_t bytes = 2 * tiles * sizeof(float);
    const cudaError_t rc = cudaMalloc((void**)&s_dMetrics, bytes);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] cudaMalloc metrics tiles=%zu rc=%d", tiles, (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }
    s_tilesCap = tiles;
    return true;
}

static bool ensureHostPinned(size_t tiles) {
    if (tiles <= s_hostCap && s_hPinned) return true;
    if (s_hPinned) { (void)cudaFreeHost(s_hPinned); s_hPinned = nullptr; }
    s_hostCap = 0;

    const size_t bytes = 2 * tiles * sizeof(float);
    const cudaError_t rc = cudaHostAlloc((void**)&s_hPinned, bytes, cudaHostAllocPortable);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][WARN] cudaHostAlloc pinned slab failed tiles=%zu rc=%d (falling back to pageable)", tiles, (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }
    s_hostCap = tiles;
    return true;
}

// -------------------------------- public API --------------------------------
namespace HeatmapMetrics {

bool buildGPU(RendererState& state,
              int width, int height, int tilePx,
              cudaStream_t stream) noexcept
{
    if (width <= 0 || height <= 0 || tilePx <= 0) return false;
    if (!state.d_iterations.get()) return false;

    const int px = std::max(1, tilePx);
    const int tilesX = (width  + px - 1) / px;
    const int tilesY = (height + px - 1) / px;
    const size_t tiles = (size_t)tilesX * (size_t)tilesY;

    if (!ensureDeviceBuffers(tiles)) return false;
    const bool havePinned = ensureHostPinned(tiles);

    float* dBoundary = s_dMetrics;           // Device: Channel 0
    float* dContrast = s_dMetrics + tiles;   // Device: Channel 1

    // Nacktmull: single-thread-per-tile for deterministic accumulation
    dim3 grid((unsigned)tilesX, (unsigned)tilesY, 1);
    dim3 block(1, 1, 1);

    kernel_tile_metrics<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(state.d_iterations.get()),
        width, height, px, tilesX, tilesY,
        dBoundary, dContrast
    );
    cudaError_t rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] kernel launch rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    // Host destinations: prefer pinned slab for D2H, then memcpy into vectors.
    float* hBoundary = nullptr;
    float* hContrast = nullptr;

    if (havePinned) {
        hBoundary = s_hPinned;
        hContrast = s_hPinned + tiles;
        rc = cudaMemcpyAsync(hBoundary, dBoundary, tiles * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (rc == cudaSuccess)
            rc = cudaMemcpyAsync(hContrast, dContrast, tiles * sizeof(float), cudaMemcpyDeviceToHost, stream);
    } else {
        // Fallback: direct to pageable vectors (slower D2H)
        state.h_entropy.resize(tiles);
        state.h_contrast.resize(tiles);
        hBoundary = state.h_entropy.data();
        hContrast = state.h_contrast.data();
        rc = cudaMemcpyAsync(hBoundary, dBoundary, tiles * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (rc == cudaSuccess)
            rc = cudaMemcpyAsync(hContrast, dContrast, tiles * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] memcpyAsync metrics->host rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    rc = cudaStreamSynchronize(stream);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] streamSync metrics rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    // If we used pinned host, mirror into vectors now (stable ordering).
    if (havePinned) {
        state.h_entropy.resize(tiles);
        state.h_contrast.resize(tiles);
        std::memcpy(state.h_entropy.data(),  hBoundary, tiles * sizeof(float));
        std::memcpy(state.h_contrast.data(), hContrast, tiles * sizeof(float));
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM][GPU][NACKTMULL] ok tiles=%dx%d N=%zu tilePx=%d pinned=%d",
                       tilesX, tilesY, tiles, px, havePinned ? 1 : 0);
    }
    return true;
}

} // namespace HeatmapMetrics
