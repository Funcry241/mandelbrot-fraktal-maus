// Datei: src/renderer_loop.cpp
// Zeilen: 109
// 🐭 Maus-Kommentar: Implementiert die zentralen Frame-Operationen für das Rendering. `renderFrame()` ist der neue API-Einstiegspunkt, intern leitet er an `renderFrame_impl()` weiter. Schneefuchs sagt: „Formell sauber, Otter-kompatibel.“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"  // ✅ drawFullscreenQuad()

namespace RendererLoop {

void beginFrame(RendererState& state) {
    double currentTime = glfwGetTime(); // Sekunden
    state.deltaTime = static_cast<float>(currentTime - state.lastTime); // Sekunden
    state.lastTime = currentTime;

    state.frameCount++;
    if (state.deltaTime > 0.0f) {
        state.currentFPS = 1.0f / state.deltaTime;
    }
}

void updateTileSize(RendererState& state) {
    // Dynamische Anpassung der Tile-Größe könnte hier erfolgen
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

    Hud::draw(state);  // 🆕 Neue vereinfachte Signatur: nur noch `RendererState&`

    glfwSwapBuffers(state.window);
}

void renderFrame_impl(RendererState& state, bool autoZoomEnabled) {
    beginFrame(state);
    updateTileSize(state);
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
