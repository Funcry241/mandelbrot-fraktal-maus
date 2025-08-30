// Loop orchestrates the FramePipeline deterministically; periodic device-log flush, clear dt tracking.
// Otter: No duplicate upload/draw here – the pipeline does it. ASCII logs, compact. (Bezug zu Otter)
// Schneefuchs: C4127-free via if constexpr; removed unnecessary includes/statics. (Bezug zu Schneefuchs)
// New (Otter/Schneefuchs): 60 FPS cap via FrameLimiter + optional VSync — smooth pacing, low jitter.
// New (Otter): Ultra-low-overhead capture of the 100th frame via async PBO+fence (single-shot, no stall).

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "frame_pipeline.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "frame_limiter.hpp"   // header in src/
#include "frame_capture.hpp"   // async single-shot 100th-frame capture (Otter)
#include <cuda_runtime_api.h>

namespace RendererLoop {

namespace {
    static pace::FrameLimiter g_frameLimiter; // namespace pace
    static bool g_vsyncInit = false;

    static inline void beginFrameLocal(RendererState& state) {
        const double now = glfwGetTime();
        double delta = now - state.lastTime;
        if (delta < 0.0) delta = 0.0;
        state.deltaTime = static_cast<float>(delta);
        state.lastTime  = now;
        state.frameCount++; // 1-based after first frame
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

    // Full frame pipeline (CUDA → Upload → Draw → Overlays → PERF)
    FramePipeline::execute(state);

    // --- Single-shot async capture of the 100th frame (non-blocking) -----------
    // Cost per frame is a tiny branch. Actual GPU readback is enqueued once,
    // then we poll completion on later frames without stalling the render path.
    FrameCapture::OnFrameRendered(state.frameCount);

    // Device log flush (debug or periodic)
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

    // 60 FPS cap — precise sleep+spin pacing, low jitter.
    if constexpr (Settings::capFramerate) {
        g_frameLimiter.limit(Settings::capTargetFps);
    } else {
        g_frameLimiter.limit(0); // update internals without sleeping
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
