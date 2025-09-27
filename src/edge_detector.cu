///// Otter: CUDA edge-detector — compact kernel; cross-gradient (±R) per grid sample.
///// Schneefuchs: Numeric rc logs; deterministic behavior; no GL; ASCII-only.
///// Maus: One kernel launch; async D2H; stream sync; host argmax; tiny footprint.
//// Datei: src/edge_detector.cu

#include "pch.hpp"
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "edge_detector.cuh"
#include "renderer_state.hpp"
#include "luchs_cuda_log_buffer.hpp"

namespace {

// 1D clamp
__device__ __forceinline__ int clampi(int v, int a, int b) {
    return v < a ? a : (v > b ? b : v);
}

// Jede Thread-Instanz bewertet genau EIN Raster-Sample.
__global__ void kernel_edge_scores(const uint16_t* __restrict__ it,
                                   int w, int h,
                                   int samplesX, int samplesY,
                                   int probeR,
                                   float* __restrict__ outScore,
                                   int2*  __restrict__ outPos)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = samplesX * samplesY;
    if (tid >= total) return;

    const int sx = tid % samplesX;
    const int sy = tid / samplesX;

    // gleichmäßig über Bild, zentriert in jeder Zelle
    const float fx = (sx + 0.5f) / (float)samplesX;
    const float fy = (sy + 0.5f) / (float)samplesY;
    int px = (int)(fx * (float)w);
    int py = (int)(fy * (float)h);
    px = clampi(px, 0, w - 1);
    py = clampi(py, 0, h - 1);

    // Finite Differences (Kreuz, Abstand probeR)
    const int xm = clampi(px - probeR, 0, w - 1);
    const int xp = clampi(px + probeR, 0, w - 1);
    const int ym = clampi(py - probeR, 0, h - 1);
    const int yp = clampi(py + probeR, 0, h - 1);

    const int idx_xm = py * w + xm;
    const int idx_xp = py * w + xp;
    const int idx_ym = ym * w + px;
    const int idx_yp = yp * w + px;

    const float gx = fabsf((float)it[idx_xp] - (float)it[idx_xm]);
    const float gy = fabsf((float)it[idx_yp] - (float)it[idx_ym]);

    // L2-Gradient (selektiver als L1).
    const float score = sqrtf(gx*gx + gy*gy);

    outScore[tid] = score;
    outPos[tid]   = make_int2(px, py);
}

} // anon

namespace EdgeDetector {

bool findStrongestEdge(RendererState& state,
                       int width, int height,
                       int samplesX, int samplesY,
                       int probeRadiusPx,
                       cudaStream_t stream,
                       Result& out) noexcept
{
    out = Result{};

    if (width <= 0 || height <= 0) return false;
    if (samplesX <= 0 || samplesY <= 0) return false;
    if (probeRadiusPx <= 0) probeRadiusPx = 1;
    if (!state.d_iterations.get()) return false;

    const int total = samplesX * samplesY;

    // Device temporaries
    float* d_scores = nullptr;
    int2*  d_pos    = nullptr;

    cudaError_t rc = cudaSuccess;
    rc = cudaMalloc((void**)&d_scores, (size_t)total * sizeof(float));
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[EDGE][ERR] cudaMalloc scores n=%d rc=%d", total, (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }
    rc = cudaMalloc((void**)&d_pos, (size_t)total * sizeof(int2));
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[EDGE][ERR] cudaMalloc pos n=%d rc=%d", total, (int)rc);
        cudaFree(d_scores);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    // Launch
    const int threads = 128;
    const int blocks  = (total + threads - 1) / threads;
    kernel_edge_scores<<<blocks, threads, 0, stream>>>(
        static_cast<const uint16_t*>(state.d_iterations.get()),
        width, height,
        samplesX, samplesY,
        probeRadiusPx,
        d_scores, d_pos
    );
    rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[EDGE][ERR] launch rc=%d", (int)rc);
        cudaFree(d_scores); cudaFree(d_pos);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    // Copy back
    std::vector<float> h_scores((size_t)total);
    std::vector<int2>  h_pos((size_t)total);

    rc = cudaMemcpyAsync(h_scores.data(), d_scores, (size_t)total * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess) {
        rc = cudaMemcpyAsync(h_pos.data(), d_pos, (size_t)total * sizeof(int2),
                             cudaMemcpyDeviceToHost, stream);
    }
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[EDGE][ERR] memcpyAsync rc=%d", (int)rc);
        cudaFree(d_scores); cudaFree(d_pos);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    rc = cudaStreamSynchronize(stream);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[EDGE][ERR] streamSync rc=%d", (int)rc);
        cudaFree(d_scores); cudaFree(d_pos);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    // Argmax auf Host (tiny)
    int   bestIx = -1;
    float bestSc = -1.0f;
    for (int i = 0; i < total; ++i) {
        const float s = h_scores[(size_t)i];
        if (s > bestSc) { bestSc = s; bestIx = i; }
    }

    cudaFree(d_scores);
    cudaFree(d_pos);

    if (bestIx < 0) return false;

    out.bestPx = h_pos[(size_t)bestIx].x;
    out.bestPy = h_pos[(size_t)bestIx].y;
    out.grad   = bestSc;  // vereinheitlicht: 'grad'

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[EDGE] best px=(%d,%d) grad=%.3f samples=%dx%d R=%d",
                       out.bestPx, out.bestPy, (double)out.grad,
                       samplesX, samplesY, probeRadiusPx);
    }
    return true;
}

} // namespace EdgeDetector
