// Datei: src/renderer_loop.cpp
// Zeilen: 185
// 🐭 Maus-Kommentar: Sanftes, zoom-skalierbares Zielsystem – jetzt wechselt der Otter das Ziel auch bei winzigen Fraktalverschiebungen im tiefsten Zoom. Schneefuchs: „Große Würfe erkennt man auch am leisesten Zucken.“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"

namespace RendererLoop {

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

void updateTileSize(RendererState& state) {
    int newSize = computeTileSizeFromZoom(state.zoom);
    if (newSize != state.lastTileSize || state.lastTileSize == 0) {
        state.lastTileSize = newSize;

        OpenGLUtils::setGLResourceContext("tileSizeChange");
        state.resize(state.width, state.height);

        if (Settings::debugLogging) {
            std::printf("[DEBUG] TileSize changed -> resize to %dx%d with tileSize=%d\n", state.width, state.height, newSize);
        }
    }
}

void computeCudaFrame(RendererState& state) {
    float2 newOffset = {};  // ⚠️ defensiv initialisiert
    bool shouldZoom = false;

    CudaInterop::renderCudaFrame(
        state.d_iterations,
        state.d_entropy,
        state.width,
        state.height,
        static_cast<double>(state.zoom),
        make_double2(state.offset.x, state.offset.y),
        state.maxIterations,
        state.h_entropy,
        newOffset,
        shouldZoom,
        state.lastTileSize,
        state
    );


    RendererPipeline::updateTexture(state.pbo, state.tex, state.width, state.height);

    state.shouldZoom = shouldZoom;

    if (shouldZoom) {
        double dx = static_cast<double>(newOffset.x) - state.targetOffset.x;
        double dy = static_cast<double>(newOffset.y) - state.targetOffset.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        double dynamicJumpThreshold = Settings::MIN_JUMP_DISTANCE / state.zoom;
        if (dist > dynamicJumpThreshold) {
            state.updateOffsetTarget(newOffset);
            if (Settings::debugLogging) {
                std::printf("[DEBUG] Target updated | Δ=%.3e > threshold=%.3e\n", dist, dynamicJumpThreshold);
            }
        } else {
            if (Settings::debugLogging) {
                std::printf("[DEBUG] Target ignored | Δ=%.3e <= threshold=%.3e\n", dist, dynamicJumpThreshold);
            }
        }
    }
}

void updateAutoZoom(RendererState& state) {
    static float2 lastTarget = {0.0f, 0.0f};

    if (!state.shouldZoom) return;

    bool newTarget = std::fabs(state.targetOffset.x - lastTarget.x) > 1e-12f ||
                     std::fabs(state.targetOffset.y - lastTarget.y) > 1e-12f;

    state.zoom *= Settings::AUTOZOOM_SPEED;

    float2 delta = {
        state.targetOffset.x - static_cast<float>(state.offset.x),
        state.targetOffset.y - static_cast<float>(state.offset.y)
    };

    float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    if (dist < Settings::DEADZONE) {
        if (newTarget && Settings::debugLogging) {
            std::printf("[DEBUG] Target reached: delta=%.3e < DEADZONE (%.1e)\n", dist, Settings::DEADZONE);
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
        std::printf("[DEBUG] New target tile: targetOffset = (%.10f, %.10f)\n", state.targetOffset.x, state.targetOffset.y);
        std::printf("[DEBUG] Δ=%.3e | dist=%.6f | tanh=%.3f | move=(%.3e, %.3e)\n", dist, dist, rawTanh, moveX, moveY);
        std::printf("[DEBUG] New offset = (%.10f, %.10f)\n", state.offset.x, state.offset.y);
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
