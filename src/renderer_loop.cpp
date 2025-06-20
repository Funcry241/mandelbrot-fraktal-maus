// Datei: src/renderer_loop.cpp
// Zeilen: 124
// 🐭 Maus-Kommentar: Haupt-Frame-Loop mit CUDA-Interop, dynamischer Tile-Größe, Auto-Zoom & HUD. Jetzt mit sauberer PBO-/Textur-Erzeugung via OpenGLUtils. Schneefuchs: „Modularisieren wie ein Otter seinen Bau – sonst undicht!“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"  // 🆕 PBO & Textur über Helper erzeugen

namespace RendererLoop {

void initResources(RendererState& state) {
    // 🔧 OpenGL-PBO & Textur erzeugen via Helper
    state.pbo = OpenGLUtils::createPBO(state.width, state.height);
    state.tex = OpenGLUtils::createTexture(state.width, state.height);

    // ⚙️ CUDA-Interop aktivieren
    CudaInterop::registerPBO(state.pbo);

    // 🎨 HUD initialisieren
    Hud::init();

    if (Settings::debugLogging) {
        std::puts("[DEBUG] initResources() abgeschlossen");
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

void updateTileSize(RendererState& state) {
    state.lastTileSize = computeTileSizeFromZoom(state.zoom);
}

void computeCudaFrame(RendererState& state) {
    float2 newOffset;
    bool shouldZoom = false;

    CudaInterop::renderCudaFrame(
        state.d_iterations,
        state.d_entropy,
        state.width,
        state.height,
        state.zoom,
        state.offset,
        state.maxIterations,
        state.h_entropy,
        newOffset,
        shouldZoom,
        state.lastTileSize
    );

    state.shouldZoom = shouldZoom;
    state.targetOffset = newOffset;
}

void updateAutoZoom(RendererState& state) {
    if (state.shouldZoom) {
        state.zoom *= Settings::AUTOZOOM_SPEED;
        state.offset.x = state.offset.x * (1.0f - Settings::LERP_FACTOR) + state.targetOffset.x * Settings::LERP_FACTOR;
        state.offset.y = state.offset.y * (1.0f - Settings::LERP_FACTOR) + state.targetOffset.y * Settings::LERP_FACTOR;
    }
}

void drawFrame(RendererState& state) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    RendererPipeline::drawFullscreenQuad(state.tex);
    Hud::draw(state);

    glfwSwapBuffers(state.window);
}

void renderFrame_impl(RendererState& state, bool autoZoomEnabled) {
    beginFrame(state);
    updateTileSize(state);

    state.adaptIterationCount();  // 🧠 Iterationsanzahl dynamisch anpassen – Schneefuchs war hier streng!

    computeCudaFrame(state);

    if (autoZoomEnabled) {
        updateAutoZoom(state);
    }

    drawFrame(state);
}

void renderFrame(RendererState& state, bool autoZoomEnabled) {
    renderFrame_impl(state, autoZoomEnabled);
}

} // namespace RendererLoop
