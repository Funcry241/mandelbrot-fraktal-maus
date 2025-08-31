///// MAUS
///// OWNER
///// RESERVED
///// Datei: src/renderer_core.cu

///// Otter: Renderer-Core – GL-Init/Window, Loop-Delegation; keine Zoom-Logik hier.
///// Schneefuchs: CUDA/GL strikt getrennt; deterministische ASCII-Logs; Ressourcen klar besitzend.
///// Maus: Alpha 80 – Pipeline/Loop entscheidet; Renderer zeichnet/tauscht nur, ohne Doppelpfad.

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
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

    // createWindow() macht den Kontext bereits current; zweiter Aufruf ist harmlos.
    glfwMakeContextCurrent(state.window);
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGL context made current");
        if (glfwGetCurrentContext() != state.window) {
            LUCHS_LOG_HOST("[ERROR] Current OpenGL context is not the GLFW window!");
        } else {
            LUCHS_LOG_HOST("[CHECK] OpenGL context correctly bound to window");
        }
    }

    // GLEW dynamisch initialisieren (Projektpolicy). Für Core-Profiles nötig:
    glewExperimental = GL_TRUE;
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

    // Low-level Pipeline (FSQ/Shader etc.)
    RendererPipeline::init();
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] RendererPipeline initialized");

    // GL-Resourcen + CUDA-Interop
    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");
        state.pbo.initAsPixelBuffer(state.width, state.height);
        // Texture via Utils (immutable storage). Wrapper bleibt Eigentümer des Handles.
        state.tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(state.width, state.height));

        // CUDA-Interop an PBO koppeln
        CudaInterop::registerPBO(state.pbo);

        // Saubere GL-State: PBO unbinden
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

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

    // Ab hier übernimmt die Loop die komplette Frame-Pipeline (Upload & Draw)
    RendererLoop::renderFrame_impl(this->state);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Returned from RendererLoop::renderFrame_impl");

    // Kein doppelter Draw – die Pipeline zeichnet bereits.
    glfwSwapBuffers(state.window);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DRAW] glfwSwapBuffers called");

#if ENABLE_ZOOM_LOGGING
    LUCHS_LOG_HOST("[ZOOM] post-frame: zoom=%.6f center=(%.6f,%.6f)",
                   state.zoom, state.center.x, state.center.y);
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
    // WICHTIG: Alle GL-Objekte löschen, solange ein gültiger Kontext existiert!
    RendererPipeline::cleanup();
    WarzenschweinOverlay::cleanup();
    HeatmapOverlay::cleanup();

    // CUDA-Interop freigeben (benötigt keinen GL-Kontext)
    CudaInterop::unregisterPBO();

    // Fenster (und damit GL-Kontext) zerstören
    RendererWindow::destroyWindow(state.window);

    // Host-Seite
    freeDeviceBuffers();
    glfwTerminate();

    glInitialized = false;
    glResourcesInitialized = false;
}
