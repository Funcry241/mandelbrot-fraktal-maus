///// Otter: Renderer-Core – GL-Init/Window, Loop-Delegation; keine Zoom-Logik hier.
///// Schneefuchs: CUDA/GL strikt getrennt; deterministische ASCII-Logs; Ressourcen klar besitzend.
///// Maus: Alpha 80 – Pipeline/Loop entscheidet; Renderer zeichnet/tauscht nur, ohne Doppelpfad.
///// Datei: src/renderer_core.cu

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "heatmap_overlay.hpp"
#include "frame_pipeline.hpp"
#include "luchs_log_host.hpp"   // <-- Logging-Makro verfügbar machen

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>

Renderer::Renderer(int width, int height)
: state(width, height)
{
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] Renderer::Renderer() started");
    }
}

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::initGL() {
    if (glInitialized) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[WARN] Renderer::initGL() called twice; ignoring");
        }
        return true;
    }

    // Fenster + GL-Kontext
    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        LUCHS_LOG_HOST("[ERROR] Failed to create GLFW window");
        return false;
    }

    // createWindow() macht den Kontext bereits current; zweiter Aufruf ist harmlos.
    glfwMakeContextCurrent(state.window);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGL context made current");
        if (glfwGetCurrentContext() != state.window) {
            LUCHS_LOG_HOST("[ERROR] Current OpenGL context is not the GLFW window!");
        } else {
            LUCHS_LOG_HOST("[CHECK] OpenGL context correctly bound to window");
        }
    }

    // GLEW init (idempotent genug; Fehler sauber loggen)
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        LUCHS_LOG_HOST("[FATAL] glewInit failed: %s", reinterpret_cast<const char*>(glewGetErrorString(glewErr)));
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        return false;
    }

    // Swap-Interval NICHT mehr an Settings koppeln (Settings::vsync fehlt im Build).
    // Wir lassen den aktuell gesetzten GLFW-Default unangetastet.

    // GL-Ressourcen anlegen: PBO + Texture (immutable storage)
    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");
        state.pbo = Hermelin::GLBuffer(OpenGLUtils::createPBO(state.width, state.height));
        state.tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(state.width, state.height));

        // PBO bei CUDA registrieren (CUDA-13 kompatible Registrierung in CudaInterop)
        CudaInterop::registerPBO(state.pbo);

        // Saubere GL-State: PBO unbinden
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glResourcesInitialized = true;
        if constexpr (Settings::debugLogging) {
            GLint boundPBO = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPBO);
            LUCHS_LOG_HOST("[CHECK] initGL - GL PBO bound: %d | PBO ID: %u", boundPBO, state.pbo.id());
        }
    }

    glInitialized = true;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[INIT] GL init complete (w=%d h=%d)", state.width, state.height);
    }

    // GPU-Pipeline vorbereiten
    RendererPipeline::init();

    // Device-Buffer anlegen (tileSize bleibt in Pipeline/Zoom-Logik konfiguriert)
    state.setupCudaBuffers(Settings::BASE_TILE_SIZE > 0 ? Settings::BASE_TILE_SIZE : 16);

    return true;
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::renderFrame() {
    // Delegation an Pipeline: komplette Reihenfolge (CUDA → Analyse → Upload → Draw → Logs)
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] Entering Renderer::renderFrame");
    }

    // Frame ausführen (kein Device-wide Sync in diesem Pfad)
    FramePipeline::execute(state);

    // Fenster zeigen / Events pumpen
    glfwSwapBuffers(state.window);
    glfwPollEvents();
}

void Renderer::freeDeviceBuffers() {
    // GPU/GL Buffers
    state.d_iterations.free();
    state.d_entropy.free();
    state.d_contrast.free();

    state.pbo.free();
    state.tex.free();
}

void Renderer::resize(int newW, int newH) {
    if (newW <= 0 || newH <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[WARN] Renderer::resize ignored invalid size %dx%d", newW, newH);
        }
        return;
    }
    if (newW == state.width && newH == state.height) return;

    // State passt intern PBO/Tex/Device-Buffer an und registriert PBO ggf. neu.
    state.resize(newW, newH);

    // Swap-Interval nicht anfassen; vorhandene GLFW-Einstellung bleibt bestehen.
}

void Renderer::cleanup() {
    if (!glInitialized) return;

    // GL abhängig: Pipeline zuerst
    RendererPipeline::cleanup();
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
