// MAUS:
// Datei: src/renderer_core.cu
// 🐭 Maus-Kommentar: Alpha 80 – Kontextfix & klare Zuständigkeiten. Keine Zoom-Logik mehr hier.
// 🦦 Otter: Renderer steuert nur Fenster/GL; Zoom/Analyse liegen in der Pipeline/Loop. (Bezug zu Otter)
// 🦊 Schneefuchs: CUDA und OpenGL sauber getrennt, keine verwaisten Referenzen. (Bezug zu Schneefuchs)
#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_resources.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
// #include "zoom_logic.hpp"      // Schneefuchs: Zoom-Entscheidung liegt nicht mehr hier
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "luchs_log_host.hpp"
#include "renderer_loop.hpp"

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
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGL context made current");
        if (glfwGetCurrentContext() != state.window) {
            LUCHS_LOG_HOST("[ERROR] Current OpenGL context is not the GLFW window!");
        } else {
            LUCHS_LOG_HOST("[CHECK] OpenGL context correctly bound to window");
        }
    }

    if (glewInit() != GLEW_OK) {
        LUCHS_LOG_HOST("[ERROR] glewInit() failed");
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        glfwTerminate();
        return false;
    }

    if (Settings::debugLogging) {
        const GLubyte* version = glGetString(GL_VERSION);
        LUCHS_LOG_HOST("[CHECK] OpenGL version: %s", version ? reinterpret_cast<const char*>(version) : "unknown");
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[CHECK] glGetError after context init = 0x%04X", err);
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[CHECK] glPixelStorei set GL_UNPACK_ALIGNMENT = 1");

    RendererPipeline::init();
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] RendererPipeline initialized");

    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");
        state.pbo.initAsPixelBuffer(state.width, state.height);
        state.tex.create();
        CudaInterop::registerPBO(state.pbo);
        glResourcesInitialized = true;
        if (Settings::debugLogging) {
            GLint boundPBO = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPBO);
            LUCHS_LOG_HOST("[CHECK] initGL - GL PBO bound: %d | PBO ID: %u", boundPBO, state.pbo.id());
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

void Renderer::renderFrame() {
    glClear(GL_COLOR_BUFFER_BIT);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] glClear called");

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Entering Renderer::renderFrame");

    // Ab hier übernimmt die Loop die komplette Frame-Pipeline (inkl. Upload & Draw)
    RendererLoop::renderFrame_impl(this->state);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Returned from RendererLoop::renderFrame_impl");

    // 🐑 Schneefuchs: Doppelte Upload/Draw entfernt – die Pipeline zeichnet bereits.
    // (Früher: OpenGLUtils::updateTextureFromPBO(...) + glDrawArrays(...) -> redundant)

    glfwSwapBuffers(state.window);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DRAW] glfwSwapBuffers called");

    // ⚠️ Alte Zoom-Logik entfernt:
    // - Kein Zugriff mehr auf state.zoomResult
    // - Keine Verwendung von Settings::zoomFactor
    // Zoom/Offset werden in RendererLoop/FramePipeline aktualisiert.
#if ENABLE_ZOOM_LOGGING
    {
        // Optional: hier könnten HUD/Debug-Infos über state.warzenschweinText geloggt werden.
        LUCHS_LOG_HOST("[ZOOM] post-frame: zoom=%.6f offset=(%.6f,%.6f)",
                       state.zoom, state.offset.x, state.offset.y);
    }
#endif
}

void Renderer::freeDeviceBuffers() {
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
        LUCHS_LOG_HOST("[CHECK] resize - GL PBO bound: %d | PBO ID: %u", boundPBO, state.pbo.id());
    }
}

void Renderer::cleanup() {
    RendererPipeline::cleanup();
    CudaInterop::unregisterPBO();
    RendererWindow::destroyWindow(state.window);
    WarzenschweinOverlay::cleanup();
    freeDeviceBuffers();
    HeatmapOverlay::cleanup();
    glfwTerminate();
    glInitialized = false;
}
