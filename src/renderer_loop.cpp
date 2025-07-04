// Datei: src/renderer_loop.cpp
// Zeilen: 318
// 🐭 Maus-Kommentar: Projekt Capybara Phase 2 - Konsistente Heatmap-Daten-Fluss mit Null-Initialisierung.
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

    // Context aktualisieren
    ctx.width = state.width;
    ctx.height = state.height;
    ctx.maxIterations = state.maxIterations;
    ctx.tileSize = state.lastTileSize;
    ctx.supersampling = state.supersampling;
    ctx.d_iterations = state.d_iterations;
    ctx.d_entropy = state.d_entropy;
    ctx.d_contrast = state.d_contrast;
    ctx.h_entropy = state.h_entropy;
    ctx.h_contrast = state.h_contrast;
    ctx.overlayActive = state.overlayEnabled;
    ctx.lastEntropy = state.lastEntropy;
    ctx.lastContrast = state.lastContrast;
    ctx.lastTileIndex = state.lastTileIndex;

    beginFrame(state);
    computeCudaFrame(ctx, state);

    // Capybara: Buffers før Entropie-/Kontrastberechnung nullen
    {
        size_t tilesX = (ctx.width + ctx.tileSize - 1) / ctx.tileSize;
        size_t tilesY = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
        size_t tilesCount = tilesX * tilesY;
        CUDA_CHECK(cudaMemset(ctx.d_entropy, 0, tilesCount * sizeof(float)));
        CUDA_CHECK(cudaMemset(ctx.d_contrast, 0, tilesCount * sizeof(float)));
    }

    // Capybara: Heatmap-Daten berechnen
    CudaInterop::computeCudaEntropyContrast(
        ctx.d_iterations,
        ctx.d_entropy,
        ctx.d_contrast,
        ctx.width,
        ctx.height,
        ctx.tileSize,
        ctx.maxIterations
    );

    // Capybara: Device->Host kopieren
    {
        size_t tilesX = (ctx.width + ctx.tileSize - 1) / ctx.tileSize;
        size_t tilesY = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
        size_t tilesCount = tilesX * tilesY;
        CUDA_CHECK(cudaMemcpy(ctx.h_entropy.data(), ctx.d_entropy, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ctx.h_contrast.data(), ctx.d_contrast, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));
    }

    if (Settings::debugLogging) {
        float e0 = ctx.h_entropy.empty() ? 0.0f : ctx.h_entropy[0];
        float c0 = ctx.h_contrast.empty() ? 0.0f : ctx.h_contrast[0];
        std::printf("[Heatmap] Entropy[0]=%.4f Contrast[0]=%.4f\n", e0, c0);
    }

    // Draw fractal
    RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height);
    drawFrame(ctx, state.tex, state);

    // Capybara: Heatmap-Overlay direkt aufrufen (Entropie + Kontrast)
    HeatmapOverlay::drawOverlay(
        ctx.h_entropy,
        ctx.h_contrast,
        ctx.width,
        ctx.height,
        ctx.tileSize,
        0,       // textureId unused
        state    // aktueller RendererState
    );

    Hud::draw(state);

    // State zurückschreiben
    state.zoom = ctx.zoom;
    state.offset = ctx.offset;
    state.h_entropy = ctx.h_entropy;
    state.h_contrast = ctx.h_contrast;
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
