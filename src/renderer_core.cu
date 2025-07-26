// 🐭 Maus-Kommentar: Alpha 49e - Supersampling vollständig entfernt. Kein d_tileSupersampling, kein h_tileSupersampling, kein Overhead mehr. Otter: "Sauberer, schneller, schlanker." Schneefuchs zufrieden.

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_loop.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"

#define ENABLE_ZOOM_LOGGING 0

Renderer::Renderer(int width, int height)
: state(width, height), glInitialized(false) {
    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] Renderer::Renderer() started");
}

Renderer::~Renderer() {
    if (Settings::debugLogging && !glInitialized)
        LUCHS_LOG("[DEBUG] Cleanup skipped - OpenGL not initialized");

    if (glInitialized) {
        if (Settings::debugLogging)
            LUCHS_LOG("[DEBUG] Calling cleanup()");
        cleanup();
    }
}

bool Renderer::initGL() {
    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] initGL() called");

    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        LUCHS_LOG("[ERROR] Failed to create GLFW window");
        return false;
    }

    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] GLFW window created successfully");

    glfwMakeContextCurrent(state.window);
    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] OpenGL context made current");

    if (glewInit() != GLEW_OK) {
        LUCHS_LOG("[ERROR] glewInit() failed");
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        glfwTerminate();
        return false;
    }

    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] GLEW initialized successfully");

    RendererPipeline::init();
    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] RendererPipeline initialized");

    glInitialized = true;
    if (Settings::debugLogging)
        LUCHS_LOG("[DEBUG] OpenGL init complete");

    return true;
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::renderFrame_impl() {
    RendererLoop::renderFrame_impl(state);

    if (state.zoomResult.isNewTarget) {
        state.lastEntropy  = state.zoomResult.bestEntropy;
        state.lastContrast = state.zoomResult.bestContrast;
    }

    state.offset = state.zoomResult.newOffset;

    if (state.zoomResult.shouldZoom)
        state.zoom *= Settings::zoomFactor;

#if ENABLE_ZOOM_LOGGING
    const auto& zr = state.zoomResult;
    const float2& off = state.offset;
    const float2& tgt = zr.newOffset;
    float dx = tgt.x - off.x;
    float dy = tgt.y - off.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    static float2 lastTarget = { 0.0f, 0.0f };
    static int stayCounter = 0;
    bool jumped = (tgt.x != lastTarget.x || tgt.y != lastTarget.y);
    if (jumped) {
        stayCounter = 0;
        lastTarget = tgt;
    } else {
        stayCounter++;
    }

    LUCHS_LOG(
        "[ZoomLog] Z=%.5e Idx=%3d E=%.4f C=%.4f dE=%.4f dC=%.4f Dist=%.6f Thresh=%.6f RelE=%.3f RelC=%.3f New=%d Stay=%d\n",
        state.zoom, zr.bestIndex,
        zr.bestEntropy, zr.bestContrast,
        zr.bestEntropy - state.lastEntropy, zr.bestContrast - state.lastContrast,
        zr.distance, zr.minDistance,
        zr.relEntropyGain, zr.relContrastGain,
        zr.isNewTarget ? 1 : 0, stayCounter
    );
#endif
}

void Renderer::freeDeviceBuffers() {
    if (state.d_iterations) { CUDA_CHECK(cudaFree(state.d_iterations)); state.d_iterations = nullptr; }
    if (state.d_entropy)    { CUDA_CHECK(cudaFree(state.d_entropy));    state.d_entropy = nullptr; }
    if (state.d_contrast)   { CUDA_CHECK(cudaFree(state.d_contrast));   state.d_contrast = nullptr; }

    state.h_entropy.clear();
    state.h_contrast.clear();
}

void Renderer::resize(int newW, int newH) {
    LUCHS_LOG("[INFO] Resize: %d x %d\n", newW, newH);
    state.resize(newW, newH);
    glViewport(0, 0, newW, newH);
}

void Renderer::cleanup() {
    RendererPipeline::cleanup();
    CudaInterop::unregisterPBO();

    glDeleteBuffers(1, &state.pbo);
    glDeleteTextures(1, &state.tex);

    RendererWindow::destroyWindow(state.window);
    WarzenschweinOverlay::cleanup();

    freeDeviceBuffers();
    HeatmapOverlay::cleanup();

    glfwTerminate();
    glInitialized = false;
}
