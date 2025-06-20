// Datei: src/renderer_loop.cpp
// Zeilen: 150
// 🐭 Maus-Kommentar: Haupt-Frame-Loop mit dynamischem GPU-Resize, geglättetem Auto-Zoom & Texture-Hygiene via Helferfunktion. Schneefuchs: „Wer PBOs von Hand pflegt, macht sich schmutzig. Nutze deine Werkzeuge!“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"

namespace RendererLoop {

// 🧵 Framebuffer-Callback zur Reaktion auf Fenstergröße
static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    using namespace CudaInterop;
    if (globalRendererState && width > 0 && height > 0) {
        globalRendererState->resize(width, height);
    }
}

void initResources(RendererState& state) {
    if (state.pbo != 0 || state.tex != 0) {
        if (Settings::debugLogging) {
            std::puts("[DEBUG] initResources() übersprungen – Ressourcen bereits initialisiert");
        }
        return;
    }

    state.pbo = OpenGLUtils::createPBO(state.width, state.height);
    state.tex = OpenGLUtils::createTexture(state.width, state.height);

    CudaInterop::registerPBO(state.pbo);
    Hud::init();
    state.setupCudaBuffers();

    glfwSetFramebufferSizeCallback(state.window, framebufferSizeCallback);

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

    RendererPipeline::updateTexture(state.pbo, state.tex, state.width, state.height);

    state.shouldZoom = shouldZoom;
    state.targetOffset = newOffset;
}

void updateAutoZoom(RendererState& state) {
    static float2 lastTarget = {0.0f, 0.0f};

    if (!state.shouldZoom) return;

    bool newTarget = std::fabs(state.targetOffset.x - lastTarget.x) > 1e-12f ||
                     std::fabs(state.targetOffset.y - lastTarget.y) > 1e-12f;

    state.zoom *= Settings::AUTOZOOM_SPEED;

    float2 delta = {
        state.targetOffset.x - state.offset.x,
        state.targetOffset.y - state.offset.y
    };

    float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    if (dist < Settings::DEADZONE) {
        if (newTarget && Settings::debugLogging) {
            std::printf("[DEBUG] ▶ Ziel erreicht: delta=%.3e < DEADZONE (%.1e)\n", dist, Settings::DEADZONE);
        }
        return;
    }

    float rawTanh = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    float factor = Settings::my_clamp(rawTanh * Settings::MAX_OFFSET_FRACTION, 0.0f, 1.0f);

    float moveX = delta.x * factor;
    float moveY = delta.y * factor;

    state.offset.x += moveX;
    state.offset.y += moveY;

    if (newTarget && Settings::debugLogging) {
        std::printf("[DEBUG] Neues Ziel-Tile: targetOffset = (%.10f, %.10f)\n", state.targetOffset.x, state.targetOffset.y);
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
    state.adaptIterationCount();
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
