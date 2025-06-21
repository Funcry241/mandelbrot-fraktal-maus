// Datei: src/renderer_loop.cpp
// Zeilen: 130
// 🐭 Maus-Kommentar: Haupt-Frame-Loop mit CUDA-Interop, dynamischer Tile-Größe, Auto-Zoom & HUD. Jetzt mit tanh-gedämpfter Offset-Annäherung – Schneefuchs: „So flüssig wie ein Otter im Gleitflug!“

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

    // 🔁 PBO nach Texture übertragen (damit drawFullscreenQuad was sieht)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, state.pbo);
    glBindTexture(GL_TEXTURE_2D, state.tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state.width, state.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    state.shouldZoom = shouldZoom;
    state.targetOffset = newOffset;
}

void updateAutoZoom(RendererState& state) {
    static float2 lastTarget = {0.0f, 0.0f};  // 🧠 Nur ändern, wenn Ziel wechselt

    if (!state.shouldZoom) return;

    bool newTarget = std::fabs(state.targetOffset.x - lastTarget.x) > 1e-12f ||
                     std::fabs(state.targetOffset.y - lastTarget.y) > 1e-12f;

    // 📌 Zoom-Fortschritt
    state.zoom *= Settings::AUTOZOOM_SPEED;

    // ➗ Abstand zum Zieloffset berechnen
    float2 delta = {
        state.targetOffset.x - state.offset.x,
        state.targetOffset.y - state.offset.y
    };

    float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    // 🧊 Bewegung stoppen, wenn fast am Ziel
    if (dist < Settings::DEADZONE) {
        if (newTarget && Settings::debugLogging) {
            std::printf("[DEBUG] ▶ Ziel erreicht: delta=%.3e < DEADZONE (%.1e)\n", dist, Settings::DEADZONE);
        }
        return;
    }

    // 🌀 Dämpfung via tanh + Fraktionslimitierung
    float rawTanh = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    float factor = Settings::my_clamp(rawTanh * Settings::MAX_OFFSET_FRACTION, 0.0f, 1.0f);

    float moveX = delta.x * factor;
    float moveY = delta.y * factor;

    state.offset.x += moveX;
    state.offset.y += moveY;

    // 🧭 Nur loggen, wenn das Ziel neu ist (d.h. neuer Tile-Fokus)
    if (newTarget && Settings::debugLogging) {
        std::printf("[DEBUG] ▶ Neues Ziel-Tile: targetOffset = (%.10f, %.10f)\n", state.targetOffset.x, state.targetOffset.y);
        std::printf("[DEBUG] Δ=%.3e | dist=%.6f | tanh=%.3f | move=(%.3e, %.3e)\n", dist, dist, rawTanh, moveX, moveY);
        std::printf("[DEBUG] Neuer Offset = (%.10f, %.10f)\n", state.offset.x, state.offset.y);
    }

    lastTarget = state.targetOffset;
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
