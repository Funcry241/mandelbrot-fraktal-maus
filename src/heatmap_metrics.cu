///// Otter: GPU heatmap metrics — compact kernel, hash-binned entropy, stddev contrast.
///// Schneefuchs: No GL; numeric rc logs; slab device buffers; deterministic behavior.
///// Maus: One kernel launch; immediate stream sync for same-frame use.

#include "pch.hpp"
#include "heatmap_metrics.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// -------------------------------- kernel --------------------------------
__global__ void kernel_tile_metrics(const uint16_t* __restrict__ it,
                                    int w, int h,
                                    int tilePx, int tilesX, int tilesY,
                                    float* __restrict__ entropy,
                                    float* __restrict__ contrast)
{
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int x0 = tx * tilePx;
    const int y0 = ty * tilePx;
    const int x1 = min(w, x0 + tilePx);
    const int y1 = min(h, y0 + tilePx);

    const int tileW = max(0, x1 - x0);
    const int tileH = max(0, y1 - y0);
    const int nPix  = tileW * tileH;
    const int outIx = ty * tilesX + tx;

    if (nPix <= 0) {
        if (entropy)  entropy[outIx]  = 0.0f;
        if (contrast) contrast[outIx] = 0.0f;
        return;
    }

    // Ein-Pass: Summe, Summe^2 und Hash-Histogramm
    double sum = 0.0;
    double sum2 = 0.0;

    constexpr int B = 32;
    int hist[B];
    #pragma unroll
    for (int i = 0; i < B; ++i) hist[i] = 0;

    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            const int v = (int)row[x];
            sum  += (double)v;
            sum2 += (double)v * (double)v;

            const int b = (v ^ (v >> 5)) & (B - 1);
            hist[b] += 1;
        }
    }

    // Kontrast = Standardabweichung
    const double invN = 1.0 / (double)nPix;
    const double mean = sum * invN;
    double var = sum2 * invN - mean * mean;
    if (var < 0.0) var = 0.0;
    if (contrast) contrast[outIx] = (float)sqrt(var);

    // Entropie (hash-binned, 32 Buckets), log2
    float H = 0.0f;
    const float invNf  = 1.0f / (float)nPix;
    constexpr float invLn2 = 1.0f / 0.6931471805599453f;
    for (int i = 0; i < B; ++i) {
        const float p = (float)hist[i] * invNf;
        if (p > 0.0f) H -= p * (logf(p) * invLn2);
    }
    if (entropy) entropy[outIx] = H;
}

// --------------- device slab buffer for entropy+contrast ----------------
static float* s_dMetrics = nullptr;   // layout: [tiles] entropy | [tiles] contrast
static size_t s_tilesCap = 0;

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

    float* dEntropy  = s_dMetrics;
    float* dContrast = s_dMetrics + tiles;

    dim3 grid((unsigned)tilesX, (unsigned)tilesY, 1);
    dim3 block(1, 1, 1);

    kernel_tile_metrics<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(state.d_iterations.get()),
        width, height, px, tilesX, tilesY,
        dEntropy, dContrast
    );
    cudaError_t rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] kernel launch rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    state.h_entropy.resize(tiles);
    state.h_contrast.resize(tiles);

    rc = cudaMemcpyAsync(state.h_entropy.data(),  dEntropy,  tiles * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(state.h_contrast.data(), dContrast, tiles * sizeof(float),
                             cudaMemcpyDeviceToHost, stream);
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

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM][GPU] ok tiles=%dx%d N=%zu tilePx=%d", tilesX, tilesY, tiles, px);
    }
    return true;
}

} // namespace HeatmapMetrics
