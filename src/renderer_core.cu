// Datei: src/renderer_core.cu
// Zeilen: 129
// 🐭 Maus-Kommentar: Flugente aktiviert! float2 statt double2 zur Wiederherstellung der FPS. Statistik-Fix: lastEntropy/lastContrast werden jetzt *immer* gesetzt, unabhängig vom Debug-Flag. Schneefuchs: „Jetzt liefern auch Release-Builds saubere Metriken.“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_loop.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "hud.hpp"
#include "cuda_interop.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"

#define ENABLE_ZOOM_LOGGING 0

Renderer::Renderer(int width, int height)
: state(width, height), glInitialized(false) {}

Renderer::~Renderer() {
    if (Settings::debugLogging && !glInitialized) {
        std::puts("[DEBUG] cleanup() skipped – OpenGL was never initialized");
    }
    if (glInitialized) {
        cleanup();
    }
}

bool Renderer::initGL() {
    if (Settings::debugLogging) std::puts("[DEBUG] initGL aufgerufen");

    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        std::puts("[ERROR] Fenstererstellung fehlgeschlagen (GLFW)");
        return false;
    }

    if (glewInit() != GLEW_OK) {
        std::puts("[ERROR] glewInit() fehlgeschlagen");
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        // --- Otter/Schneefuchs: Jetzt auch global GLFW terminieren! ---
        glfwTerminate();
        return false;
    }

    RendererPipeline::init();

    if (Settings::debugLogging) std::puts("[DEBUG] OpenGL-Initialisierung abgeschlossen");
    glInitialized = true;
    return true;
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::renderFrame_impl(bool autoZoomEnabled) {
    RendererLoop::renderFrame_impl(state, autoZoomEnabled);

    // --- Statistik jetzt immer aktuell, unabhängig vom Logging! ---
    state.lastEntropy  = state.zoomResult.bestEntropy;
    state.lastContrast = state.zoomResult.bestContrast;

#if ENABLE_ZOOM_LOGGING
    float ox = state.offset.x;
    float oy = state.offset.y;
    float tx = state.smoothedTargetOffset.x;
    float ty = state.smoothedTargetOffset.y;
    float dx = tx - ox, dy = ty - oy;
    float dist = std::sqrt(dx * dx + dy * dy);

    static float2 lastTarget = { 0.0f, 0.0f };
    static int stayCounter = 0;
    bool jumped = (tx != lastTarget.x || ty != lastTarget.y);
    if (jumped) {
        stayCounter = 0;
        lastTarget = { tx, ty };
    } else {
        stayCounter++;
    }

    const auto& zr = state.zoomResult;
    std::printf(
        "ZoomLog Z %.5e Idx %3d Ent %.5f Ctr %.5f dE %.5f dC %.5f Dist %.6f Thresh %.6f RelE %.3f RelC %.3f New %d Stayed %d\n",
        state.zoom, zr.bestIndex, zr.bestEntropy, zr.bestContrast,
        zr.bestEntropy - state.lastEntropy, zr.bestContrast - state.lastContrast,
        zr.distance, zr.minDistance, zr.relEntropyGain, zr.relContrastGain,
        zr.isNewTarget ? 1 : 0, stayCounter
    );

    if (state.justZoomed) {
        CudaInterop::logZoomEvaluation(state.d_iterations, state.width, state.height, state.maxIterations, state.zoom);
        state.justZoomed = false;
    }
#endif
}

void Renderer::freeDeviceBuffers() {
    if (state.d_iterations) { CUDA_CHECK(cudaFree(state.d_iterations)); state.d_iterations = nullptr; }
    if (state.d_entropy) { CUDA_CHECK(cudaFree(state.d_entropy)); state.d_entropy = nullptr; }
    if (state.d_contrast) { CUDA_CHECK(cudaFree(state.d_contrast)); state.d_contrast = nullptr; }
    if (state.d_tileSupersampling){ CUDA_CHECK(cudaFree(state.d_tileSupersampling)); state.d_tileSupersampling = nullptr; }
    state.h_entropy.clear();
    state.h_contrast.clear();
    state.h_tileSupersampling.clear();
}

void Renderer::resize(int newW, int newH) {
    std::printf("[INFO] Resized to %d x %d\n", newW, newH);
    state.resize(newW, newH);
    glViewport(0, 0, newW, newH);
}

void Renderer::cleanup() {
    Hud::cleanup();
    RendererPipeline::cleanup();
    CudaInterop::unregisterPBO();

    glDeleteBuffers(1, &state.pbo);
    glDeleteTextures(1, &state.tex);

    RendererWindow::destroyWindow(state.window);

    freeDeviceBuffers();
    HeatmapOverlay::cleanup();
    glfwTerminate();

    // --- Otter/Schneefuchs: Jetzt ist alles wirklich „geputzt“! ---
    glInitialized = false;
}
