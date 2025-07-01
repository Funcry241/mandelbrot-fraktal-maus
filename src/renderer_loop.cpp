// Datei: src/renderer_loop.cpp
// Zeilen: 233
// 👝 Maus-Kommentar: Heatmap integriert! Zeigt oben rechts im Bild die Entropie- und Kontrastverteilung – live während des Auto-Zooms. Schneefuchs sagt: „Wer sehen will, was Zoom sieht, muss glühnen lassen.“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"  // ✅ Heatmap integriert
#include "frame_pipeline.hpp"   // 🧠 Neu: deterministische Frame-Steuerung
#include "zoom_command.hpp"

namespace RendererLoop {

static FrameContext ctx;
static CommandBus zoomBus;

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
    // Update Kontext-Daten aus State
    ctx.width = state.width;
    ctx.height = state.height;
    ctx.zoom = state.zoom;
    ctx.offset = state.offset;
    ctx.maxIterations = state.maxIterations;
    ctx.tileSize = state.lastTileSize;
    ctx.supersampling = state.supersampling;
    ctx.d_iterations = state.d_iterations;
    ctx.d_entropy = state.d_entropy;
    ctx.h_entropy = state.h_entropy;
    ctx.overlayActive = state.overlayEnabled;
    ctx.lastEntropy = state.lastEntropy;
    ctx.lastContrast = state.lastContrast;
    ctx.lastTileIndex = state.lastTileIndex;

    beginFrame(state);
    computeCudaFrame(ctx, state); // ✅ CUDA: Fraktal und Entropie
    RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height); // 🆕 Bild in Textur!
    if (autoZoomEnabled) applyZoomLogic(ctx, zoomBus);
    drawFrame(ctx, state.tex);

    // Synchronisiere zurück
    state.zoom = ctx.zoom;
    state.offset = ctx.offset;
    state.h_entropy = ctx.h_entropy;
    state.shouldZoom = ctx.shouldZoom;
    state.lastEntropy = ctx.lastEntropy;
    state.lastContrast = ctx.lastContrast;
    state.lastTileIndex = ctx.lastTileIndex;
}

} // namespace RendererLoop
