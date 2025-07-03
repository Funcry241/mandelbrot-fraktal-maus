// Datei: src/renderer_core.cu
// Zeilen: 127
// 🧠 Maus-Kommentar: Erweiterte Zoomlogik: Kompakter Frame-Zustandsdump inkl. Entropie-Gewinn, Kontrast-Gewinn, Distanz, Schwelle, Zielwechsel. Ideal zur Stagnationsanalyse. Schneefuchs-geeicht!

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
#include "heatmap_overlay.hpp"  // 🆕 für Cleanup-Aufruf

#define ENABLE_ZOOM_LOGGING 1  // Set to 0 to disable local zoom analysis logs

Renderer::Renderer(int width, int height)
    : state(width, height), glInitialized(false) {}

Renderer::~Renderer() {
    if (Settings::debugLogging && !glInitialized) {
        std::puts("[DEBUG] cleanup() wird trotz fehlender OpenGL-Initialisierung aufgerufen");
    }
    cleanup();
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

#if ENABLE_ZOOM_LOGGING
    double ox = state.offset.x;
    double oy = state.offset.y;
    double tx = state.smoothedTargetOffset.x;
    double ty = state.smoothedTargetOffset.y;
    double dx = tx - ox;
    double dy = ty - oy;
    double dist = std::sqrt(dx * dx + dy * dy);

    static double2 lastTarget = { 0.0, 0.0 };
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
        state.zoom,
        zr.bestIndex,
        zr.bestEntropy,
        zr.bestContrast,
        zr.bestEntropy - state.lastEntropy,
        zr.bestContrast - state.lastContrast,
        zr.distance,
        zr.minDistance,
        zr.relEntropyGain,
        zr.relContrastGain,
        zr.isNewTarget ? 1 : 0,
        stayCounter
    );

    state.lastEntropy  = zr.bestEntropy;
    state.lastContrast = zr.bestContrast;

    if (state.justZoomed) {
        CudaInterop::logZoomEvaluation(state.d_iterations, state.width, state.height, state.maxIterations, state.zoom);
        state.justZoomed = false;
    }
#endif
}

void Renderer::freeDeviceBuffers() {
    if (state.d_iterations) {
        CUDA_CHECK(cudaFree(state.d_iterations));
        state.d_iterations = nullptr;
    }
    if (state.d_entropy) {
        CUDA_CHECK(cudaFree(state.d_entropy));
        state.d_entropy = nullptr;
    }
    state.h_entropy.clear();
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

    // 🆕 Heatmap-Ressourcen freigeben
    HeatmapOverlay::cleanup();

    glfwTerminate();
}
