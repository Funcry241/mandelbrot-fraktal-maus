// ============================================================================
// Datei: src/renderer_loop.cpp
// 🐭 Maus-Kommentar: Loop orchestriert die FramePipeline deterministisch; periodisches Device-Log-Flush, klarer Δt-Track.
// 🦦 Otter: Kein doppelter Upload/Draw hier – das macht die Pipeline. ASCII-Logs, kompakt. (Bezug zu Otter)
// 🐑 Schneefuchs: C4127-frei via if constexpr; unnötige Includes & ungenutzte Statics entfernt. (Bezug zu Schneefuchs)
// Neu (Otter/Schneefuchs): 60 FPS Cap via FrameLimiter + optional VSync — smooth pacing, jitterarm.
// ============================================================================

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "frame_pipeline.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "frame_limiter.hpp"   // Header liegt in src/
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace RendererLoop {

namespace {
    // 🐭 Maus: Lokaler FrameLimiter — keine API-Änderung notwendig.
    static pace::FrameLimiter g_frameLimiter; // Namespace umbenannt von fps → pace
    static bool g_vsyncInit = false;

    // 🐑 Schneefuchs: interne Helferfunktion umbenannt, um C4211 zu vermeiden.
    static inline void beginFrameLocal(RendererState& state) {
        const float now = static_cast<float>(glfwGetTime());
        float delta = now - static_cast<float>(state.lastTime);
        if (delta < 0.0f) delta = 0.0f;                 // robust gegen Zeit-Glitches
        state.deltaTime = delta;
        state.lastTime  = static_cast<double>(now);
        state.frameCount++;
    }

    static inline void initVSyncOnce() {
        if (g_vsyncInit) return;
        g_vsyncInit = true;
        if constexpr (Settings::preferVSync) {
            glfwSwapInterval(1);
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[VSync] swapInterval=1 (tear-free)");
            }
        } else {
            glfwSwapInterval(0);
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[VSync] swapInterval=0 (software pacing only)");
            }
        }
    }
}

void renderFrame_impl(RendererState& state) {
    initVSyncOnce();
    beginFrameLocal(state);

    // Vollständige Frame-Pipeline (CUDA → Upload → Draw → Overlays → PERF)
    FramePipeline::execute(state);

    // 🐑 Schneefuchs: Device-Logs bei Bedarf flushen (Fehler oder periodisch).
    if constexpr (Settings::debugLogging) {
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess || (state.frameCount % 60 == 0)) {
            LUCHS_LOG_HOST("[Loop] flushing device logs (err=%d, frame=%d)",
                           static_cast<int>(err), state.frameCount);
            LuchsLogger::flushDeviceLogToHost(0);
        }
        if ((state.frameCount % 60) == 0) {
            LUCHS_LOG_HOST("[Loop] frame=%d dt=%.3f", state.frameCount, state.deltaTime);
        }
    }

    // 🦦 Otter: 60 FPS Cap — präzises Sleep+Spin-Pacing, jitterarm.
    if constexpr (Settings::capFramerate) {
        g_frameLimiter.limit(Settings::capTargetFps);
        // Hinweis: state.deltaTime wird am nächsten beginFrameLocal() gemessen und enthält das Pacing.
        // Dadurch bleibt der Planner „indirekt“ auf 60 FPS eingetaktet (smooth).
    } else {
        g_frameLimiter.limit(0); // update internals ohne Sleep (real dt track)
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode; (void)mods;
    if (action != GLFW_PRESS) return;

    auto* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    switch (key) {
        case GLFW_KEY_H:
            HeatmapOverlay::toggle(*state);
            break;
        case GLFW_KEY_P: {
            const bool paused = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!paused);
            break;
        }
        default:
            break;
    }
}

} // namespace RendererLoop
