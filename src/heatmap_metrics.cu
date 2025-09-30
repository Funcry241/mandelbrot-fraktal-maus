///// Datei: src/heatmap_metrics.cu
///// Otter: GPU heatmap metrics — boundary score + stddev contrast (compact).
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
// Channel 0: boundary score = pEsc * (1 - pEsc)
// Channel 1: contrast      = stddev(iterations)
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
    const int x1 = min(w, x0 + tilePx);
    const int y1 = min(h, y0 + tilePx);

    const int tileW = max(0, x1 - x0);
    const int tileH = max(0, y1 - y0);
    const int nPix  = tileW * tileH;
    const int outIx = ty * tilesX + tx;

    if (nPix <= 0) {
        if (boundaryOut) boundaryOut[outIx] = 0.0f;
        if (contrastOut) contrastOut[outIx] = 0.0f;
        return;
    }

    // Ein Pass: Summe, Summe^2, Escape-Zähler
    double sum = 0.0;
    double sum2 = 0.0;
    int    countEsc = 0; // it < maxIter  → muss hostseitig vorab so interpretiert werden

    // Hinweis:
    // Wir kennen maxIter hier nicht getrennt; die Iterationspuffer enthalten
    // nach dem Render die tatsächliche Zählung. Konvention:
    //  - "escaped" ≈ kleinere Werte (typischerweise < maxIter)
    //  - "in set"  ≈ Werte nahe/maxIter (Kernel-seitig bekannt)
    //
    // Für den Boundary-Score benötigen wir nur das Mischungsverhältnis.
    // Wir nutzen eine pragmatische Schwelle relativ zum lokalen Mittel,
    // vermeiden aber Abhängigkeit von maxIter im Kernel. Das ist deterministisch
    // und robust genug für die Kachel-Entscheidung.
    //
    // Vorgehen in zwei Schritten:
    //  (1) ersten Durchlauf: sum/sum2 + grobe Mittelwertschätzung
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

    //  (2) zweiter Durchlauf: Escape-Anteil grob per Mittelwertschwelle
    //      (praktisch: escaped-Pixel haben signifikant niedrigere it)
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

    float* dBoundary = s_dMetrics;           // Channel 0
    float* dContrast = s_dMetrics + tiles;   // Channel 1

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

    state.h_entropy.resize(tiles);   // reuse slot: now 'boundary'
    state.h_contrast.resize(tiles);

    rc = cudaMemcpyAsync(state.h_entropy.data(),  dBoundary, tiles * sizeof(float),
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
