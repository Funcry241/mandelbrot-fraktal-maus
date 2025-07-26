// Datei: src/renderer_loop.cpp
// 🐭 Maus-Kommentar: Erweiterter PerfLog – misst resize() + Swap-Zeit separat. Ermöglicht Analyse von FPS-Limitierung durch VSync oder Buffer-Recreation.
// 🦦 Otter: `renderFrame_impl` ist jetzt voll implementiert – keine Linkerleichen mehr!
// 🐑 Schneefuchs: Ressourcensicher, nachvollziehbar und ready für Release-Debugging.

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "frame_pipeline.hpp"
#include "zoom_command.hpp"
#include "zoom_logic.hpp"
#include "frame_pipeline.hpp"
#include <chrono>
#include <cmath> // für std::sqrt, std::clamp

namespace RendererLoop {

static FrameContext ctx;
static CommandBus zoomBus;
static bool isFirstFrame = true;

void initResources(RendererState& state) {
    if (state.pbo != 0 || state.tex != 0) return;

    OpenGLUtils::setGLResourceContext("init");
    state.pbo = OpenGLUtils::createPBO(state.width, state.height);
    state.tex = OpenGLUtils::createTexture(state.width, state.height);
    CudaInterop::registerPBO(state.pbo);

    state.lastTileSize = computeTileSizeFromZoom(static_cast<float>(state.zoom));
}

void beginFrame(RendererState& state) {
    float currentTime = static_cast<float>(glfwGetTime());
    float delta = currentTime - static_cast<float>(state.lastTime);
    if (delta < 0.0f) delta = 0.0f;

    state.deltaTime = delta;
    state.lastTime = static_cast<double>(currentTime);
    state.frameCount++;
}

void renderFrame_impl(RendererState& state) {
    beginFrame(state);
    initResources(state);

    FramePipeline::execute(state);

    if (Settings::debugLogging && state.frameCount % 60 == 0) {
        LUCHS_LOG("[Loop] Frame %d, Δt = %.3f\n", state.frameCount, state.deltaTime);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode; (void)mods;
    if (action != GLFW_PRESS) return;

    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    switch (key) {
        case GLFW_KEY_H:
            HeatmapOverlay::toggle(*state);
            break;
        case GLFW_KEY_P: {
            bool paused = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!paused);
            break;
        }
        default:
            break;
    }
}

} // namespace RendererLoop
