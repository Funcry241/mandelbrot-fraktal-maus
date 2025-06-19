// Datei: src/renderer_loop.cpp
// Zeilen: 121
// 🐭 Maus-Kommentar: Zentrale Frame-Schleife mit integrierter dynamischer Tile-Anpassung. `initResources()` initialisiert Textur, PBO, CUDA-Interop und HUD. `updateTileSize()` passt die Blockgröße dem Zoom an – fließend wie der Otterblick. Schneefuchs: „Wer zoomt, der fließt.“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"

namespace RendererLoop {

void initResources(RendererState& state) {
    // 🔧 OpenGL-PBO erzeugen
    glGenBuffers(1, &state.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, state.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, state.width * state.height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 🖼️ OpenGL-Textur erzeugen
    glGenTextures(1, &state.tex);
    glBindTexture(GL_TEXTURE_2D, state.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, state.width, state.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // ⚙️ CUDA-Interop aktivieren
    CudaInterop::registerPBO(state.pbo);

    // 🎨 HUD initialisieren
    Hud::init();  // ✅ Korrigiert: kein Parameter übergeben

#if defined(DEBUG) || defined(_DEBUG)
    if (Settings::debugLogging) {
        std::puts("[DEBUG] initResources() abgeschlossen");
    }
#endif
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
    int tileSize = 32;
    if (state.zoom > 30000.0f)
        tileSize = 4;
    else if (state.zoom > 3000.0f)
        tileSize = 8;
    else if (state.zoom > 1000.0f)
        tileSize = 16;

    tileSize = std::max(4, std::min(tileSize, 32));
    state.lastTileSize = tileSize;
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
