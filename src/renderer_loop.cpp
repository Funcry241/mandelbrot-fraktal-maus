// MAUS:
// Datei: src/renderer_loop.cpp
// 🐭 Maus-Kommentar: Loop orchestriert die FramePipeline deterministisch; periodisches Device-Log-Flush, klarer Δt-Track.
// 🦦 Otter: Kein doppelter Upload/Draw hier – das macht die Pipeline. ASCII-Logs, kompakt. (Bezug zu Otter)
// 🐑 Schneefuchs: C4127-frei via if constexpr; unnötige Includes & ungenutzte Statics entfernt. (Bezug zu Schneefuchs)

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
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace RendererLoop {

// 🐑 Schneefuchs: interne Helferfunktion umbenannt, um C4211 zu vermeiden.
static inline void beginFrameLocal(RendererState& state) {
    const float now = static_cast<float>(glfwGetTime());
    float delta = now - static_cast<float>(state.lastTime);
    if (delta < 0.0f) delta = 0.0f;                 // robust gegen Zeit-Glitches
    state.deltaTime = delta;
    state.lastTime  = static_cast<double>(now);
    state.frameCount++;
}

void renderFrame_impl(RendererState& state) {
    beginFrameLocal(state);

    // Vollständige Frame-Pipeline (CUDA → Upload → Draw → Overlays → PERF)
    FramePipeline::execute(state);

    // 🐑 Schneefuchs: C4127-frei – compile-time Gate statt konstantem if.
    if constexpr (Settings::debugLogging) {
        // 🦦 Otter: Device-Log bei Fehlern oder alle 60 Frames flushen.
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
