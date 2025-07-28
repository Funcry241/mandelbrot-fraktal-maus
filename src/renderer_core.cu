// Datei: src/renderer_core.cu
// 🐭 Maus-Kommentar: Alpha 67 - Kontextfix: PBO-Registrierung erst nach aktivem GL-Kontext. Kein invalid argument mehr. 
// 🦦 Otter: CUDA sieht jetzt korrekt. Fokus, keine Zufälle.
// 🦊 Schneefuchs: Reihenfolge gewahrt, Kontextfehler eliminiert.

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_loop.hpp"
#include "renderer_resources.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "luchs_log_host.hpp"

#define ENABLE_ZOOM_LOGGING 0

Renderer::Renderer(int width, int height)
: state(width, height), glInitialized(false), glResourcesInitialized(false) {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] Renderer::Renderer() started");
}

Renderer::~Renderer() {
    if (Settings::debugLogging && !glInitialized)
        LUCHS_LOG_HOST("[DEBUG] Cleanup skipped - OpenGL not initialized");

    if (glInitialized) {
        if (Settings::debugLogging)
            LUCHS_LOG_HOST("[DEBUG] Calling cleanup()");
        cleanup();
    }
}

bool Renderer::initGL() {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] initGL() called");

    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        LUCHS_LOG_HOST("[ERROR] Failed to create GLFW window");
        return false;
    }

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] GLFW window created successfully");

    glfwMakeContextCurrent(state.window);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] OpenGL context made current");

    if (glewInit() != GLEW_OK) {
        LUCHS_LOG_HOST("[ERROR] glewInit() failed");
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        glfwTerminate();
        return false;
    }

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] GLEW initialized successfully");

    RendererPipeline::init();
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] RendererPipeline initialized");

    // 🧠 Kontext ist jetzt gültig - PBO und CUDA-Interop erst ab hier
    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");
        state.pbo = OpenGLUtils::createPBO(state.width, state.height);
        CudaInterop::registerPBO(state.pbo);

        state.tex = OpenGLUtils::createTexture(state.width, state.height);
        glResourcesInitialized = true;

        if (Settings::debugLogging) {
            GLint boundPBO = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPBO);
            LUCHS_LOG_HOST("[CHECK] initGL - OpenGL PBO bound: %d | Created PBO ID: %u", boundPBO, state.pbo);
        }
    }

    glInitialized = true;
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] OpenGL init complete");

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

    LUCHS_LOG_HOST(
        "[ZoomLog] Z=%.5e Idx=%3d E=%.4f C=%.4f dE=%.4f dC=%.4f Dist=%.6f Thresh=%.6f RelE=%.3f RelC=%.3f New=%d Stay=%d",
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
    LUCHS_LOG_HOST("[INFO] Resize: %d x %d", newW, newH);
    state.resize(newW, newH);
    glViewport(0, 0, newW, newH);

    if (Settings::debugLogging) {
        GLint boundPBO = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPBO);
        LUCHS_LOG_HOST("[CHECK] resize - OpenGL PBO bound: %d | Active PBO ID: %u", boundPBO, state.pbo);
    }
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
