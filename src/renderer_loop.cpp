// Datei: src/renderer_loop.cpp
// 🐭 Maus-Kommentar: Alpha 59a – WarzenschweinOverlay zeigt nun auch dynamisch Iterationen, TileSize, AutoZoom. Alles ASCII-sicher. Schneefuchs: „Live-Telemetrie. Endlich klar.“

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "frame_pipeline.hpp"
#include "zoom_command.hpp"
#include <chrono>

namespace RendererLoop {

static FrameContext ctx;
static CommandBus zoomBus;
static bool isFirstFrame = true;

void initResources(RendererState& state) {
    if (state.pbo != 0 || state.tex != 0) return;

    OpenGLUtils::setGLResourceContext("init");
    state.pbo = OpenGLUtils::createPBO(state.width, state.height);
    state.tex = OpenGLUtils::createTexture(state.width, state.height);
    CudaInterop::registerPBO(state.pbo);

    state.lastTileSize = computeTileSizeFromZoom(static_cast<float>(state.zoom));
    state.setupCudaBuffers();
}

void beginFrame(RendererState& state) {
    float currentTime = static_cast<float>(glfwGetTime());
    float delta = currentTime - static_cast<float>(state.lastTime);
    if (delta < 0.0f) delta = 0.0f;

    state.deltaTime = delta;
    state.lastTime = static_cast<double>(currentTime);
    state.frameCount++;

    if (state.deltaTime > 0.0f)
        state.fps = 1.0f / state.deltaTime;
}

void renderFrame_impl(RendererState& state) {
    auto frameStart = std::chrono::high_resolution_clock::now();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ctx.zoom         = static_cast<float>(state.zoom);
    ctx.offset       = { static_cast<float>(state.offset.x), static_cast<float>(state.offset.y) };
    ctx.width        = state.width;
    ctx.height       = state.height;
    ctx.maxIterations= state.maxIterations;
    ctx.tileSize     = state.lastTileSize;
    ctx.supersampling= state.supersampling;
    ctx.d_iterations = state.d_iterations;
    ctx.d_entropy    = state.d_entropy;
    ctx.d_contrast   = state.d_contrast;
    ctx.h_entropy    = state.h_entropy;
    ctx.h_contrast   = state.h_contrast;
    ctx.overlayActive= state.heatmapOverlayEnabled; // nur Heatmap beeinflusst Entropielogik
    ctx.lastEntropy  = state.lastEntropy;
    ctx.lastContrast = state.lastContrast;

    if (isFirstFrame) isFirstFrame = false;

    beginFrame(state);

    size_t totalPixels = static_cast<size_t>(ctx.width) * ctx.height;
    size_t tilesX = (ctx.width + ctx.tileSize - 1) / ctx.tileSize;
    size_t tilesY = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
    size_t tilesCount = tilesX * tilesY;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (ctx.d_iterations)
        CUDA_CHECK(cudaMemset(ctx.d_iterations, 0, totalPixels * sizeof(int)));

    computeCudaFrame(ctx, state);

    auto t1 = std::chrono::high_resolution_clock::now();

    RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height);
    drawFrame(ctx, state.tex, state);

    auto t2 = std::chrono::high_resolution_clock::now();

    if (ctx.d_entropy)  CUDA_CHECK(cudaMemset(ctx.d_entropy, 0, tilesCount * sizeof(float)));
    if (ctx.d_contrast) CUDA_CHECK(cudaMemset(ctx.d_contrast, 0, tilesCount * sizeof(float)));

    CudaInterop::computeCudaEntropyContrast(
        ctx.d_iterations, ctx.d_entropy, ctx.d_contrast,
        ctx.width, ctx.height, ctx.tileSize, ctx.maxIterations
    );

    CUDA_CHECK(cudaMemcpy(ctx.h_entropy.data(), ctx.d_entropy, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_contrast.data(), ctx.d_contrast, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));

    auto t3 = std::chrono::high_resolution_clock::now();

    if (Settings::debugLogging) {
        float cudaMs    = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float drawMs    = std::chrono::duration<float, std::milli>(t2 - t1).count();
        float analyzeMs = std::chrono::duration<float, std::milli>(t3 - t2).count();
        float totalMs   = std::chrono::duration<float, std::milli>(t3 - frameStart).count();
        float e0 = ctx.h_entropy.empty() ? 0.0f : ctx.h_entropy[0];
        float c0 = ctx.h_contrast.empty() ? 0.0f : ctx.h_contrast[0];
        std::printf("[Perf] cuda=%.2fms draw=%.2fms analyze=%.2fms total=%.2fms | E=%.4f C=%.4f\n",
            cudaMs, drawMs, analyzeMs, totalMs, e0, c0);
        std::printf("[DEBUG] WarzenschweinOverlay enabled = %d\n", static_cast<int>(state.warzenschweinOverlayEnabled));
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (state.heatmapOverlayEnabled) {
        HeatmapOverlay::drawOverlay(
            ctx.h_entropy, ctx.h_contrast,
            ctx.width, ctx.height, ctx.tileSize, 0, state
        );
    }

    if (state.warzenschweinOverlayEnabled) {
        int zoomExp = static_cast<int>(std::round(std::log10(1.0 / state.zoom)));
        std::string warzText =
            "OtterDream Mandelbrot\n"
            "Zoom: 1e" + std::to_string(zoomExp) + "\n"
            "FPS:  " + std::to_string(state.fps) + "\n"
            "Iter: " + std::to_string(state.maxIterations) + "\n"
            "Tile: " + std::to_string(state.lastTileSize) + "\n"
            "Auto: " + (CudaInterop::getPauseZoom() ? "Paused" : "Active");

        WarzenschweinOverlay::setText(warzText);
        WarzenschweinOverlay::drawOverlay(state);
    }

    state.zoom         = static_cast<double>(ctx.zoom);
    state.offset       = { ctx.offset.x, ctx.offset.y };
    state.h_entropy    = ctx.h_entropy;
    state.h_contrast   = ctx.h_contrast;
    state.lastEntropy  = ctx.lastEntropy;
    state.lastContrast = ctx.lastContrast;
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
            bool paused = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!paused);
            break;
        }
        default:
            break;
    }
}

} // namespace RendererLoop
