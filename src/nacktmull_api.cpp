///// Otter: Host wrapper for CUDA render/analysis; ASCII-only logs.
///// Schneefuchs: Uses RendererState-owned render/copy streams; no globals; robust error paths.
///// Maus: Mirrors old computeCudaFrame; no API drift for callers.
///// Datei: src/nacktmull_api.cpp

#include "pch.hpp"
#include "nacktmull_api.hpp"
#include <GL/glew.h>
#include "cuda_interop.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "common.hpp"
#include "core_kernel.h"
#include "perturbation_orbit.hpp"
#include <vector_functions.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <exception>
#include <vector>
#include <cmath> // std::abs

// Device symbol for CONST-path upload (defined in core_kernel.cu)
extern __constant__ double2 zrefConst[];

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

    // Hinweis: Resize/Init registriert alle PBOs; per-frame Register ist idempotent implementiert.
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

    // Allokation + Host-Pinning sicherstellen
    state.setupCudaBuffers(fctx.tileSize);

    // Streams: bevorzugt State-Streams, deterministischer Fallback = 0
    cudaStream_t renderStrm = state.renderStream ? state.renderStream : (cudaStream_t)0;
    cudaStream_t copyStrm   = state.copyStream   ? state.copyStream   : renderStrm;

    if ((!state.renderStream || !state.copyStream) && Settings::debugLogging) {
        LUCHS_LOG_HOST("[STREAM][WARN] fallback used (render=%p, copy=%p)", (void*)renderStrm, (void*)copyStrm);
    }

    // ============================ PERT: Orbit-Upload ============================
    // Build+Upload reference orbit when active and rebase is needed.
    if (Settings::pertEnable && fctx.zoom >= Settings::pertZoomMin)
    {
        // Canonical pixel scale (screen→complex): sy = 2/height * 1/zoom; sx = sy * aspect.
        const double invZoom = (fctx.zoom != 0.0) ? (1.0 / fctx.zoom) : 1.0;
        const double ar      = (fctx.height > 0) ? double(fctx.width) / double(fctx.height) : 1.0;
        const double sy      = (fctx.height > 0) ? (2.0 / double(fctx.height)) * invZoom : 2.0 * invZoom;
        const double sx      = sy * ar;

        const double2 c_now{ double(fctx.offset.x), double(fctx.offset.y) };

        bool needRebase = (state.zrefCount <= 0);
        if (!needRebase) {
            const double dx = c_now.x - state.c_ref.x;
            const double dy = c_now.y - state.c_ref.y;
            const double dpx = std::max(std::abs(dx) / std::max(sx, 1e-300), std::abs(dy) / std::max(sy, 1e-300));
            if (dpx > Settings::deltaMaxRebase) needRebase = true;
        }

        if (needRebase) {
            try {
                std::vector<double2> orbit;
                int len = 0;
                buildReferenceOrbit(
                    c_now,
                    std::min(Settings::zrefMaxLen, state.maxIterations),
                    Settings::zrefSegSize,
                    orbit,
                    len
                );

                if (len > 0) {
                    // Choose CONST vs GLOBAL store
                    const bool preferConst = (fctx.zoom < Settings::storeSwitchZoom) && (len <= Settings::zrefMaxLen);

                    if (preferConst) {
                        // Upload to constant memory asynchronously on render stream
                        CUDA_CHECK(cudaMemcpyToSymbolAsync(
                            zrefConst, orbit.data(), size_t(len) * sizeof(double2),
                            0, cudaMemcpyHostToDevice, renderStrm));
                        // Free any previous GLOBAL buffer
                        state.freeZrefGlobal();
                        state.perturbStore = PertStore::Const;
                        // Maintain version monotonicity even for CONST path
                        state.zrefVersion += 1; if (state.zrefVersion == 0) state.zrefVersion = 1;
                    } else {
                        // Ensure GLOBAL buffer size and upload
                        state.allocateZrefGlobal(len);
                        CUDA_CHECK(cudaMemcpyAsync(
                            state.d_zrefGlobal.get(), orbit.data(),
                            size_t(len) * sizeof(double2), cudaMemcpyHostToDevice, renderStrm));
                        state.perturbStore = PertStore::Global;
                    }

                    state.c_ref       = c_now;
                    state.zrefCount   = len;
                    state.zrefSegSize = Settings::zrefSegSize;
                    state.rebaseCount += 1;

                    if constexpr (Settings::debugLogging) {
                        LUCHS_LOG_HOST("[PERT] upload ok store=%s len=%d seg=%d ver=%d zoom=%.3e",
                                       (state.perturbStore == PertStore::Const ? "CONST" : "GLOBAL"),
                                       state.zrefCount, state.zrefSegSize, state.zrefVersion, fctx.zoom);
                    }
                } else {
                    static bool s_warnedOnce = false;
                    if (!s_warnedOnce) {
                        LUCHS_LOG_HOST("[PERT][WARN] orbit empty -> fallback");
                        s_warnedOnce = true;
                    }
                    // Ensure no stale state is used
                    state.freeZrefGlobal();
                    state.zrefCount = 0;
                }
            } catch (const std::exception& ex) {
                LUCHS_LOG_HOST("[PERT][ERROR] build/upload failed: %s", ex.what());
                state.freeZrefGlobal();
                state.zrefCount = 0;
            }
        }
    } else {
        // Pert inactive at this zoom: ensure no stale GLOBAL buffer influences telemetry/launch.
        if (state.zrefCount > 0) {
            state.freeZrefGlobal();
            state.zrefCount = 0;
        }
    }

    // ======================== DEVICE RENDER (Iterations/PBO) ========================
    try {
        const float2 gpuOffset    = make_float2((float)fctx.offset.x, (float)fctx.offset.y);
        float2       gpuNewOffset = gpuOffset;

        // Renderpfad erzeugt mind. d_iterations und füllt aktuellen PBO-Slot.
        CudaInterop::renderCudaFrame(
            state.d_iterations,
            /* (E/C device buffers are handled below) */ state.d_entropy,
            state.d_contrast,
            fctx.width,
            fctx.height,
            fctx.zoom,
            gpuOffset,
            fctx.maxIterations,
            /* (host mirrors not used here) */ fctx.h_entropy,
            fctx.h_contrast,
            gpuNewOffset,
            fctx.shouldZoom,
            fctx.tileSize,
            state,
            renderStrm,
            copyStrm
        );
        (void)gpuNewOffset; // Zoom-Analyse folgt separat
    } catch (const std::exception& ex) { // [[unlikely]]
        LUCHS_LOG_HOST("[ERROR] renderCudaFrame threw: %s", ex.what());
        LuchsLogger::flushDeviceLogToHost(0);
    }

    // Unmittelbar nach Render: Fehler prüfen (nur für Logs/Flush)
    {
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) [[unlikely]] {
            LUCHS_LOG_HOST("[PIPE] Flushing device logs (err=%d)", (int)err);
            LuchsLogger::flushDeviceLogToHost(0);
        }
    }

    // ======================= ENTROPY/CONTRAST (E/C) CHAIN ==========================
    // Vollständig asynchron:
    //  - Launch beider E/C-Kernels auf renderStrm
    //  - Event am Ende der Kette (evEcDone)
    //  - copyStrm wartet auf evEcDone, dann D->H copies
    //  - evCopyDone signalisiert Host-Fertigstellung
    {
        computeCudaEntropyContrast(
            static_cast<const uint16_t*>(state.d_iterations.get()),
            static_cast<float*>(state.d_entropy.get()),
            static_cast<float*>(state.d_contrast.get()),
            fctx.width, fctx.height, fctx.tileSize, fctx.maxIterations,
            renderStrm,                   // <<< Stream
            state.evEcDone                // <<< Event nach Contrast
        );

        // Gate copies on E/C completion
        CUDA_CHECK(cudaStreamWaitEvent(copyStrm, state.evEcDone, 0));

        const size_t tilesTotal = static_cast<size_t>(numTiles);
        if (tilesTotal) {
            // D->H Kopien (Hostpuffer liegen in RendererState; pinned wenn verfügbar)
            CUDA_CHECK(cudaMemcpyAsync(
                state.h_entropy.data(),  state.d_entropy.get(),
                tilesTotal * sizeof(float), cudaMemcpyDeviceToHost, copyStrm));
            CUDA_CHECK(cudaMemcpyAsync(
                state.h_contrast.data(), state.d_contrast.get(),
                tilesTotal * sizeof(float), cudaMemcpyDeviceToHost, copyStrm));
        }

        // Signal: Host darf lesen (FramePipeline synchronisiert darauf)
        CUDA_CHECK(cudaEventRecord(state.evCopyDone, copyStrm));
    }

    // ============================= ZOOM-ANALYSE (Host) =============================
    {
        const float2 currOff = make_float2((float)fctx.offset.x, (float)fctx.offset.y);
        const float2 prevOff = currOff;

        auto zr = ZoomLogic::evaluateZoomTarget(
            state.h_entropy,                // aus RendererState
            state.h_contrast,               // aus RendererState
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
