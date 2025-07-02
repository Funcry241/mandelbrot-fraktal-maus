// Datei: src/renderer_loop.cpp
// Zeilen: 282
// 🐭 Maus-Kommentar: HeatmapOverlay wird korrekt gezeichnet – drawOverlay(ctx) ist aktiv, ctx.d_contrast ist nun gesetzt. Damit ist der vollständige Datenfluss von Kontrastwerten GPU → CPU → Heatmap sichergestellt.
#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"
#include "frame_pipeline.hpp"
#include "zoom_command.hpp"

namespace RendererLoop {

static FrameContext ctx;
static CommandBus zoomBus;
static bool isFirstFrame = true;

void initResources(RendererState& state) {
    if (state.pbo != 0 || state.tex != 0) {
        if (Settings::debugLogging) {
            std::puts("[DEBUG] initResources() skipped - resources already initialized");
        }
        return;
    }

    OpenGLUtils::setGLResourceContext("init");
    state.pbo = OpenGLUtils::createPBO(state.width, state.height);
    state.tex = OpenGLUtils::createTexture(state.width, state.height);

    CudaInterop::registerPBO(state.pbo);
    Hud::init();

    state.lastTileSize = computeTileSizeFromZoom(state.zoom);
    state.setupCudaBuffers();

    if (Settings::debugLogging) {
        std::puts("[DEBUG] initResources() completed");
    }
}

void beginFrame(RendererState& state) {
    double currentTime = glfwGetTime();
    state.deltaTime = static_cast<float>(currentTime - state.lastTime);
    state.lastTime = currentTime;

    state.frameCount++;
    if (state.deltaTime > 0.0f) {
        state.currentFPS = 1.0f / state.deltaTime;
    }
}

void renderFrame_impl(RendererState& state, bool autoZoomEnabled) {
    if (isFirstFrame) {
        ctx.zoom = state.zoom;
        ctx.offset = state.offset;
        isFirstFrame = false;
    }

    ctx.width = state.width;
    ctx.height = state.height;
    ctx.maxIterations = state.maxIterations;
    ctx.tileSize = state.lastTileSize;
    ctx.supersampling = state.supersampling;
    ctx.d_iterations = state.d_iterations;
    ctx.d_entropy = state.d_entropy;
    ctx.d_contrast = state.d_contrast;  // ✅ NEU: Kontrastdaten setzen
    ctx.h_entropy = state.h_entropy;
    ctx.overlayActive = state.overlayEnabled;
    ctx.lastEntropy = state.lastEntropy;
    ctx.lastContrast = state.lastContrast;
    ctx.lastTileIndex = state.lastTileIndex;

    beginFrame(state);
    computeCudaFrame(ctx, state);

    if (autoZoomEnabled) {
        applyZoomLogic(ctx, zoomBus);
        computeCudaFrame(ctx, state);
    }

    RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height);
    drawFrame(ctx, state.tex, state);
    drawOverlay(ctx);
    Hud::draw(state);

    state.zoom = ctx.zoom;
    state.offset = ctx.offset;
    state.h_entropy = ctx.h_entropy;
    state.shouldZoom = ctx.shouldZoom;
    state.lastEntropy = ctx.lastEntropy;
    state.lastContrast = ctx.lastContrast;
    state.lastTileIndex = ctx.lastTileIndex;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    switch (key) {
        case GLFW_KEY_H:
            HeatmapOverlay::toggle(*state);
            break;
        case GLFW_KEY_P:
            CudaInterop::setPauseZoom(!CudaInterop::getPauseZoom());
            break;
        default:
            break;
    }
}

} // namespace RendererLoop
