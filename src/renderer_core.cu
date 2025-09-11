///// Otter: Renderer-Core – GL-Init/Window, Loop-Delegation; keine Zoom-Logik hier.
///// Schneefuchs: CUDA/GL strikt getrennt; deterministische ASCII-Logs; Ressourcen klar besitzend.
///// Maus: Progressive-Cooldown + Tatze 7 Soft-Invalidate bei Sichtsprung (nach Pipeline, ohne Memset).
///// Datei: src/renderer_core.cu

#include "pch.hpp"

#include "nacktmull_api.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "cuda_interop.hpp"
#include "heatmap_overlay.hpp"
#include "frame_pipeline.hpp"
#include "luchs_log_host.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <algorithm> // std::max
#include <cmath>     // std::abs

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

    // GL-Ressourcen anlegen: PBO + Texture (immutable storage)
    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");
        state.pbo = Hermelin::GLBuffer(OpenGLUtils::createPBO(state.width, state.height));
        state.tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(state.width, state.height));

        // PBO bei CUDA registrieren
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

    // *** Progressive-Resume: __constant__-State pro Frame setzen (Cooldown beachten) ***
    {
        const bool inCooldown = (state.progressiveCooldownFrames > 0);
        const bool progReady =
            Settings::progressiveEnabled &&
            state.progressiveEnabled &&
            (state.d_stateZ.get()  != nullptr) &&
            (state.d_stateIt.get() != nullptr) &&
            !inCooldown;

        const int addIter = Settings::progressiveAddIter; // Budget pro Frame
        const int iterCap = state.maxIterations;
        const int enabled = progReady ? 1 : 0;

        nacktmull_set_progressive(
            state.d_stateZ.get(),
            state.d_stateIt.get(),
            addIter,
            iterCap,
            enabled
        );

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PROG] set_progressive enabled=%d cooldown=%d addIter=%d cap=%d z=%p it=%p",
                           enabled, state.progressiveCooldownFrames, addIter, iterCap,
                           (void*)state.d_stateZ.get(), (void*)state.d_stateIt.get());
        }
    }

    // Frame ausführen (kein Device-wide Sync in diesem Pfad)
    FramePipeline::execute(state);

    // ---------- Tatze 7: Soft-Invalidate bei Sichtsprung (nach Pipeline) ----------
    // Prüfe, ob center/zoom sprunghaft geändert wurden → 1-Frame-Pause (kein memset).
    {
        struct PrevView { bool have=false; double zoom=0.0, cx=0.0, cy=0.0; };
        static PrevView prev;

        bool justInvalidated = false;

        if (prev.have) {
            const double z0 = std::max(prev.zoom, 1e-30);
            const double z1 = std::max((double)state.zoom, 1e-30);
            const double zoomRatio = (z1 > z0) ? (z1 / z0) : (z0 / z1);

            // Pixel-Shift in Screen-Space
            const double psx = std::max(std::abs(state.pixelScale.x), 1e-30);
            const double psy = std::max(std::abs(state.pixelScale.y), 1e-30);
            const double dxPix = std::abs(((double)state.center.x - prev.cx) / psx);
            const double dyPix = std::abs(((double)state.center.y - prev.cy) / psy);
            const double panPix = std::max(dxPix, dyPix);

            const bool jump = (zoomRatio >= 1.5) || (panPix >= 8.0);

            if (jump) {
                state.invalidateProgressiveState(/*hardReset=*/false);
                justInvalidated = true;
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[PROG] soft-invalidate: zoomRatio=%.3f panPix=%.2f (th=1.5/8)",
                                   zoomRatio, panPix);
                }
            }
        }

        // Cooldown herunterzählen – aber NICHT, wenn wir eben invalidiert haben.
        if (state.progressiveCooldownFrames > 0) {
            if (!justInvalidated) {
                --state.progressiveCooldownFrames;
                if (state.progressiveCooldownFrames == 0) {
                    state.progressiveEnabled = true;
                    if constexpr (Settings::debugLogging) {
                        LUCHS_LOG_HOST("[PROG] cooldown finished → progressiveEnabled=true");
                    }
                }
            } else {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[PROG] cooldown kept at %d (fresh invalidate this frame)",
                                   state.progressiveCooldownFrames);
                }
            }
        }

        // Prev-State aktualisieren (am Ende des Frames)
        prev.zoom = (double)state.zoom;
        prev.cx   = (double)state.center.x;
        prev.cy   = (double)state.center.y;
        prev.have = true;
    }
    // ---------- Ende Tatze 7 ------------------------------------------------------

    // Fenster zeigen / Events pumpen
    glfwSwapBuffers(state.window);
    glfwPollEvents();
}

void Renderer::freeDeviceBuffers() {
    // GPU/GL Buffers
    state.d_iterations.free();
    state.d_entropy.free();
    state.d_contrast.free();

    // Progressive-State ebenfalls freigeben
    state.d_stateZ.free();
    state.d_stateIt.free();

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
