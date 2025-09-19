///// Otter: Main loop; Silk-Lite motion + frame budget pacing.
///// Schneefuchs: Device/host logs separated; flush on CUDA error paths.
///// Maus: Warm-up freeze; fixed cadence for stats; one line per event.
///// Datei: src/renderer_loop.cpp

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "frame_pipeline.hpp"
#include "cuda_interop.hpp"          // pause toggle in keyCallback
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp" // LuchsLogger::flushDeviceLogToHost
#include "heatmap_overlay.hpp"       // HeatmapOverlay::toggle
#include "frame_limiter.hpp"         // pace::FrameLimiter
#include "frame_capture.hpp"         // async single-shot 100th-frame capture
#include "warzenschwein_overlay.hpp" // WarzenschweinOverlay::toggle()
#include <cuda_runtime_api.h>        // cudaPeekAtLastError

namespace RendererLoop {

namespace {
    // Halte die Kadenz für Loop-Logs/Flush identisch zur PERF-Kadenz in frame_pipeline.cpp
    // (dort: constexpr int PERF_LOG_EVERY = 30;)
    constexpr int PERF_LOG_EVERY = 30;

    inline void beginFrameLocal(RendererState& state) {
        const double now = glfwGetTime();
        double delta = now - state.lastTime;
        if (delta < 0.0) delta = 0.0;
        // Angleichen an FramePipeline: clamp auf >= 1 ms für stabile Ableitungen/FPS
        state.deltaTime = static_cast<float>(delta < 0.001 ? 0.001 : delta);
        state.lastTime  = now;
        state.frameCount++; // 1-based after first frame
    }

    inline void initVSyncOnce() {
        static bool vsyncInit = false;
        if (vsyncInit) return;
        vsyncInit = true;
        if constexpr (Settings::preferVSync) {
            glfwSwapInterval(1);
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[VSync] swapInterval=1");
            }
        } else {
            glfwSwapInterval(0);
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[VSync] swapInterval=0");
            }
        }
    }
}

void renderFrame_impl(RendererState& state) {
    initVSyncOnce();
    beginFrameLocal(state);

    // Full frame pipeline (CUDA -> Upload -> Draw -> Overlays -> PERF)
    FramePipeline::execute(state);

    // Async single-shot capture every 100th frame (non-blocking hook)
    if ((state.frameCount % 100) == 0) {
        FrameCapture::OnFrameRendered(state.frameCount);
    }

    // Device log flush (error-triggered OR periodic with the same cadence as PERF logs)
    if constexpr (Settings::debugLogging) {
        const cudaError_t err = cudaPeekAtLastError();
        const bool periodic = (state.frameCount % PERF_LOG_EVERY) == 0;
        if (err != cudaSuccess || periodic) {
            LUCHS_LOG_HOST("[Loop] flushing device logs (err=%d, frame=%d)",
                           static_cast<int>(err), state.frameCount);
            LuchsLogger::flushDeviceLogToHost(0);
        }
        if (periodic) {
            LUCHS_LOG_HOST("[Loop] frame=%d dt=%.3f", state.frameCount, state.deltaTime);
        }
    }

    // 60 FPS cap — precise sleep+spin pacing, low jitter.
    static pace::FrameLimiter limiter;
    if constexpr (Settings::capFramerate) {
        limiter.limit(Settings::capTargetFps);
    } else {
        limiter.limit(0); // update internals without sleeping
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
        case GLFW_KEY_O:
            WarzenschweinOverlay::toggle();
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
