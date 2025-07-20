// Datei: src/renderer_loop.cpp
// Zeilen: 344
// 🐭 Maus-Kommentar: Alpha 49 – Rückgabewert von `getPauseZoom()` wird nun explizit verarbeitet, um [[nodiscard]] zu erfüllen. Kein Logikverlust, keine Warnungen. Schneefuchs: "Wer ignoriert, verliert."

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
#include <chrono>

namespace RendererLoop {

static FrameContext ctx;
static CommandBus zoomBus;
static bool isFirstFrame = true;

// 🔧 Initialisiert PBO, Texture, Cuda-Interop und HUD
void initResources(RendererState& state) {
    if (state.pbo != 0 || state.tex != 0) {
        if (Settings::debugLogging)
            std::puts("[DEBUG] initResources() skipped - resources already initialized");
        return;
    }
    OpenGLUtils::setGLResourceContext("init");
    state.pbo = OpenGLUtils::createPBO(state.width, state.height);
    state.tex = OpenGLUtils::createTexture(state.width, state.height);

    CudaInterop::registerPBO(state.pbo);
    Hud::init();

    state.lastTileSize = computeTileSizeFromZoom(static_cast<float>(state.zoom));
    state.setupCudaBuffers();

    if (Settings::debugLogging)
        std::puts("[DEBUG] initResources() completed");
}

// ⏱️ Misst Delta-Time und FPS pro Frame
void beginFrame(RendererState& state) {
    float currentTime = static_cast<float>(glfwGetTime());
    float lastTimeFloat = static_cast<float>(state.lastTime);

    float delta = currentTime - lastTimeFloat;
    if (delta < 0.0f) delta = 0.0f;  // Zeitrücksprung-Schutz

    state.deltaTime = delta;
    state.lastTime = static_cast<double>(currentTime); // state.lastTime bleibt double

    state.frameCount++;
    if (state.deltaTime > 0.0f)
        state.currentFPS = 1.0f / state.deltaTime;
}

// 🎮 Hauptschleife: Fraktal zeichnen, Analyse durchführen, HUD und Overlay anzeigen
void renderFrame_impl(RendererState& state) {
    auto frameStart = std::chrono::high_resolution_clock::now();

    ctx.zoom = static_cast<float>(state.zoom);
    ctx.offset.x = static_cast<float>(state.offset.x);
    ctx.offset.y = static_cast<float>(state.offset.y);

    if (isFirstFrame) {
        isFirstFrame = false;
    }

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

    size_t totalPixels = static_cast<size_t>(ctx.width) * ctx.height;
    size_t tilesX = (ctx.width + ctx.tileSize - 1) / ctx.tileSize;
    size_t tilesY = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
    size_t tilesCount = tilesX * tilesY;

    auto t0 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemset(ctx.d_iterations, 0, totalPixels * sizeof(int)));
    computeCudaFrame(ctx, state);

    auto t1 = std::chrono::high_resolution_clock::now();

    RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height);
    drawFrame(ctx, state.tex, state);

    auto t2 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemset(ctx.d_entropy, 0, tilesCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx.d_contrast, 0, tilesCount * sizeof(float)));
    CudaInterop::computeCudaEntropyContrast(
        ctx.d_iterations,
        ctx.d_entropy,
        ctx.d_contrast,
        ctx.width,
        ctx.height,
        ctx.tileSize,
        ctx.maxIterations
    );
    CUDA_CHECK(cudaMemcpy(ctx.h_entropy.data(), ctx.d_entropy, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_contrast.data(), ctx.d_contrast, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));

    auto t3 = std::chrono::high_resolution_clock::now();

    if (Settings::debugLogging) {
        float e0 = ctx.h_entropy.empty() ? 0.0f : ctx.h_entropy[0];
        float c0 = ctx.h_contrast.empty() ? 0.0f : ctx.h_contrast[0];
        float cudaMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float drawMs = std::chrono::duration<float, std::milli>(t2 - t1).count();
        float analysisMs = std::chrono::duration<float, std::milli>(t3 - t2).count();
        float totalMs = std::chrono::duration<float, std::milli>(t3 - frameStart).count();
        std::printf("[Perf] cuda=%.2fms draw=%.2fms analyze=%.2fms total=%.2fms\n", cudaMs, drawMs, analysisMs, totalMs);
        std::printf("[Heatmap] Entropy[0]=%.4f Contrast[0]=%.4f\n", e0, c0);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (Settings::debugLogging)
        std::puts("[HUD] glEnable(GL_BLEND) set");

    HeatmapOverlay::drawOverlay(
        ctx.h_entropy, ctx.h_contrast,
        ctx.width, ctx.height, ctx.tileSize, 0, state
    );

    Hud::draw(state);

    if (Settings::debugLogging)
        std::puts("[HUD] Hud::draw() called");

    state.zoom = static_cast<double>(ctx.zoom);
    state.offset.x = static_cast<float>(ctx.offset.x);
    state.offset.y = static_cast<float>(ctx.offset.y);
    state.h_entropy = ctx.h_entropy;
    state.h_contrast = ctx.h_contrast;
    state.shouldZoom = ctx.shouldZoom;
    state.lastEntropy = ctx.lastEntropy;
    state.lastContrast = ctx.lastContrast;
    state.lastTileIndex = ctx.lastTileIndex;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode; (void)mods;
    if (action != GLFW_PRESS) return;
    RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    switch (key) {
    case GLFW_KEY_H:
        HeatmapOverlay::toggle(*state);
        break;
    case GLFW_KEY_P: {
        // 🐭 Maus-Fix: Rückgabewert von [[nodiscard]]-Funktion nicht ignorieren!
        bool paused = CudaInterop::getPauseZoom();
        CudaInterop::setPauseZoom(!paused);
        break;
    }
    default:
        break;
    }
}

} // namespace RendererLoop
