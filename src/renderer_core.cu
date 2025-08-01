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

    // Bind OpenGL context to window
    glfwMakeContextCurrent(state.window);
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGL context made current");
        // Check context binding
        if (glfwGetCurrentContext() != state.window) {
            LUCHS_LOG_HOST("[ERROR] Current OpenGL context is not the GLFW window!");
        } else {
            LUCHS_LOG_HOST("[CHECK] OpenGL context correctly bound to window");
        }
    }

    // Initialize GLEW and verify
    if (glewInit() != GLEW_OK) {
        LUCHS_LOG_HOST("[ERROR] glewInit() failed");
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        glfwTerminate();
        return false;
    }
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] GLEW initialized successfully");
        const GLubyte* version = glGetString(GL_VERSION);
        LUCHS_LOG_HOST("[CHECK] OpenGL version: %s", version ? reinterpret_cast<const char*>(version) : "unknown");
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[CHECK] glGetError after context init = 0x%04X", err);
    }

    RendererPipeline::init();
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] RendererPipeline initialized");

    // 🧠 Kontext ist jetzt gültig - PBO und CUDA-Interop erst ab hier
    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");

        state.pbo.initAsPixelBuffer(state.width, state.height);
        state.tex.create();
        CudaInterop::registerPBO(state.pbo);

        glResourcesInitialized = true;

        if (Settings::debugLogging) {
            GLint boundPBO = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPBO);
            LUCHS_LOG_HOST("[CHECK] initGL - OpenGL PBO bound: %d | Created PBO ID: %u", boundPBO, state.pbo.id());
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
    // 🚀 Pipeline Logging
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Entering renderFrame_impl");

    // GPU Rendering
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Calling RendererLoop::renderFrame_impl");
    RendererLoop::renderFrame_impl(state);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Returned from RendererLoop::renderFrame_impl");

    // Debug: Prepare for draw
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DRAW] About to draw fullscreen quad");
    }
    
    // Zeichnen (Fullscreen Quad)
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    if (Settings::debugLogging) {
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[DRAW] glDrawArrays glGetError() = 0x%04X", err);
        LUCHS_LOG_HOST("[DRAW] Fullscreen quad drawn, swapping buffers");
    }
    
    // Swap
    glfwSwapBuffers(state.window);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DRAW] glfwSwapBuffers called");

    // State Update
    if (state.zoomResult.isNewTarget) {
        state.lastEntropy  = state.zoomResult.bestEntropy;
        state.lastContrast = state.zoomResult.bestContrast;
    }
    state.offset = state.zoomResult.newOffset;
    if (state.zoomResult.shouldZoom)
        state.zoom *= Settings::zoomFactor;

#if ENABLE_ZOOM_LOGGING
    // ... existing zoom logging ...
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
        LUCHS_LOG_HOST("[CHECK] resize - OpenGL PBO bound: %d | Active PBO ID: %u", boundPBO, state.pbo.id());
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
