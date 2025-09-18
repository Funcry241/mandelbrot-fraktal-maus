///// Otter: Host wrapper for CUDA render/analysis; ASCII-only logs.
/// ///// Schneefuchs: No global side-effects; TU-local CUDA events; robust error paths.
/// ///// Maus: Mirrors old computeCudaFrame body; no API drift for callers.
/// ///// Datei: src/nacktmull_api.cpp

#include "pch.hpp"
#include "nacktmull_api.hpp"

#include <GL/glew.h>            // GLsync, glClientWaitSync, glDeleteSync
#include "cuda_interop.hpp"      // CudaInterop::{registerPBO,renderCudaFrame}
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"        // evaluateZoomTarget(...)
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "common.hpp"            // CUDA_CHECK

#include <vector_functions.h>    // make_float2
#include <cuda_runtime.h>
#include <algorithm>
#include <exception>

namespace {

// CUDA 13: Events statt device-weitem Sync (TU-lokal)
static cudaEvent_t g_evStart = nullptr;
static cudaEvent_t g_evStop  = nullptr;
static bool        g_evInit  = false;

inline void ensureCudaEvents() {
    if (!g_evInit) {
        CUDA_CHECK(cudaEventCreate(&g_evStart));
        CUDA_CHECK(cudaEventCreate(&g_evStop));
        g_evInit = true;
    }
}

} // anon

namespace NacktmullAPI
{

void computeCudaFrame(FrameContext& fctx, RendererState& state)
{
    // [ZK] Non-blocking PBO-Ring Auswahl mit GLsync-Fences
    int tried = 0;
    const int R = RendererState::kPboRingSize;
    int next = (state.pboIndex + 1) % R;
    bool found = false;
    for (; tried < R; ++tried) {
        GLsync& fence = state.pboFence[next];
        if (fence) {
            const GLenum s = glClientWaitSync(fence, 0, 0); // poll only
            if (s == GL_ALREADY_SIGNALED || s == GL_CONDITION_SATISFIED) {
                glDeleteSync(fence); fence = 0;
            } else {
                next = (next + 1) % R; continue;
            }
        }
        found = true; break;
    }
    if (!found) {
        state.skipUploadThisFrame = true;
        if constexpr (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[ZK][UP] skip reason=no_free_pbo ring=%d", state.pboIndex);
        }
        return;
    }

    state.pboIndex = next;
    state.skipUploadThisFrame = false;

    // Jetzt den freien Slot der CUDA-Seite zuordnen
    CudaInterop::registerPBO(state.currentPBO());

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] computeCudaFrame: dimensions=%dx%d, zoom=%.5f, tileSize=%d",
                       fctx.width, fctx.height, fctx.zoom, fctx.tileSize);
    }

    const int tilesX   = (fctx.width  + fctx.tileSize - 1) / fctx.tileSize;
    const int tilesY   = (fctx.height + fctx.tileSize - 1) / fctx.tileSize;
    const int numTiles = tilesX * tilesY;

    if (fctx.tileSize <= 0 || numTiles <= 0) [[unlikely]] {
        LUCHS_LOG_HOST("[FATAL] computeCudaFrame: invalid tileSize=%d or numTiles=%d",
                       fctx.tileSize, numTiles);
        return;
    }

    state.setupCudaBuffers(fctx.tileSize);

    if constexpr (Settings::debugLogging) {
        const size_t totalPixels         = size_t(fctx.width) * size_t(fctx.height);
        const size_t need_it_bytes       = totalPixels * sizeof(uint16_t);
        const size_t need_entropy_bytes  = size_t(numTiles) * sizeof(float);
        const size_t need_contrast_bytes = size_t(numTiles) * sizeof(float);
        LUCHS_LOG_HOST("[SANITY] tiles=%d (%d x %d) pixels=%zu need(it=%zu entropy=%zu contrast=%zu) alloc(it=%zu entropy=%zu contrast=%zu)",
                       numTiles, tilesX, tilesY, totalPixels,
                       need_it_bytes, need_entropy_bytes, need_contrast_bytes,
                       state.d_iterations.size(), state.d_entropy.size(), state.d_contrast.size());
    }

    // Device-Render (Iterations -> Shade) + E/C-Analyse (erzeugt Host-Arrays)
    try {
        ensureCudaEvents();

        // GPU-Zeitmessung ohne deviceweite Synchronisation:
        CUDA_CHECK(cudaEventRecord(g_evStart, /*stream=*/0));

        float2 gpuOffset    = make_float2((float)fctx.offset.x, (float)fctx.offset.y);
        float2 gpuNewOffset = gpuOffset;

        CudaInterop::renderCudaFrame(
            state.d_iterations,
            state.d_entropy,
            state.d_contrast,
            fctx.width,
            fctx.height,
            fctx.zoom,
            gpuOffset,
            fctx.maxIterations,
            fctx.h_entropy,
            fctx.h_contrast,
            gpuNewOffset,
            fctx.shouldZoom,
            fctx.tileSize,
            state
        );

        // Stop-Event wird nach allen Kernel-Launches in Stream 0 eingereiht:
        CUDA_CHECK(cudaEventRecord(g_evStop, /*stream=*/0));
        CUDA_CHECK(cudaEventSynchronize(g_evStop)); // wartet nur auf Stream 0

        float msGpuF = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&msGpuF, g_evStart, g_evStop));
        const double msGpu = static_cast<double>(msGpuF);

        if constexpr (Settings::debugLogging) {
            if (state.lastTimings.valid) {
                LUCHS_LOG_HOST(
                    "[FRAME] Mandelbrot=%.3f | Launch=%.3f | Sync=%.3f | Entropy=%.3f | Contrast=%.3f | LogFlush=%.3f | PBOMap=%.3f",
                    state.lastTimings.mandelbrotTotal,
                    state.lastTimings.mandelbrotLaunch,
                    state.lastTimings.mandelbrotSync,
                    state.lastTimings.entropy,
                    state.lastTimings.contrast,
                    state.lastTimings.deviceLogFlush,
                    state.lastTimings.pboMap
                );
            } else {
                LUCHS_LOG_HOST("[TIME] CUDA stream0 elapsed: %.3f ms", msGpu);
            }
        }
    } catch (const std::exception& ex) { // [[unlikely]]
        LUCHS_LOG_HOST("[ERROR] renderCudaFrame threw: %s", ex.what());
        LuchsLogger::flushDeviceLogToHost(0);
    }

    // Device-Logs: nur bei Fehler sofort spülen (periodisch: am Frameende)
    {
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) [[unlikely]] {
            LUCHS_LOG_HOST("[PIPE] Flushing device logs (err=%d)", (int)err);
            LuchsLogger::flushDeviceLogToHost(0);
        }
    }

    // Zoom-Analyse (nur Hostdaten h_entropy/h_contrast)
    {
        const float2 currOff = make_float2((float)fctx.offset.x, (float)fctx.offset.y);
        const float2 prevOff = currOff;

        auto zr = ZoomLogic::evaluateZoomTarget(
            fctx.h_entropy,
            fctx.h_contrast,
            tilesX, tilesY,
            fctx.width, fctx.height,
            currOff, fctx.zoom,
            prevOff,
            state.zoomV3State
        );

        if (zr.bestIndex >= 0) {
            fctx.lastEntropy  = zr.bestEntropy;
            fctx.lastContrast = zr.bestContrast;
        } else [[unlikely]] {
            fctx.lastEntropy  = 0.0f;
            fctx.lastContrast = 0.0f;
        }

        fctx.shouldZoom = zr.shouldZoom;
        if (zr.shouldZoom) {
            fctx.newOffset = { zr.newOffset.x, zr.newOffset.y };
        }

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] ZOOMV3: best=%d score=%.3f accept=%d newOff=(%.6f,%.6f)",
                           zr.bestIndex, zr.bestScore, zr.shouldZoom ? 1 : 0,
                           zr.newOffset.x, zr.newOffset.y);
        }
    }
}

} // namespace NacktmullAPI
