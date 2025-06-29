// Datei: src/renderer_loop.cpp
// Zeilen: 228
// 👝 Maus-Kommentar: Heatmap integriert! Zeigt oben rechts im Bild die Entropie- und Kontrastverteilung – live während des Auto-Zooms. Schneefuchs sagt: „Wer sehen will, was Zoom sieht, muss glühnen lassen.“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"  // ✅ Heatmap integriert

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
    double2 newOffset = make_double2(0.0, 0.0);
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
        state.updateOffsetTarget(newOffset);
        if (Settings::debugLogging) {
            std::printf("[DEBUG] Target updated to (%.10f, %.10f)\n", newOffset.x, newOffset.y);
        }
    }
}

void updateAutoZoom(RendererState& state) {
    if (!state.shouldZoom) return;

    state.zoom *= Settings::AUTOZOOM_SPEED;

    double2 delta = {
        state.targetOffset.x - state.offset.x,
        state.targetOffset.y - state.offset.y
    };

    double dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    if (dist < Settings::DEADZONE) {
        if (Settings::debugLogging) {
            std::printf("[DEBUG] Target reached: delta=%.3e < DEADZONE (%.1e)\n", dist, Settings::DEADZONE);
        }
        return;
    }

    double rawTanh = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    double factor = Settings::my_clamp(rawTanh * Settings::MAX_OFFSET_FRACTION, 0.0, 1.0);

    double moveX = delta.x * factor;
    double moveY = delta.y * factor;

    if (Settings::debugLogging) {
        std::printf("[ZoomMove] Z=%.2e  dist=%.2e  move=(%.2e, %.2e)  factor=%.3f  → target=(%.10f, %.10f)\n",
            state.zoom,
            dist,
            moveX,
            moveY,
            factor,
            state.targetOffset.x,
            state.targetOffset.y
        );
    }

    state.offset.x += moveX;
    state.offset.y += moveY;
    state.justZoomed = true;

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Zoom update: offset -> (%.10f, %.10f)\n", state.offset.x, state.offset.y);
    }
}

void drawFrame(RendererState& state) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    RendererPipeline::drawFullscreenQuad(state.tex);

    if (!state.h_entropy.empty() && !state.zoomResult.perTileContrast.empty()) {
        HeatmapOverlay::drawOverlay(
            state.h_entropy,
            state.zoomResult.perTileContrast,
            state.width,
            state.height,
            state.lastTileSize,
            state.tex
        );
    }

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
