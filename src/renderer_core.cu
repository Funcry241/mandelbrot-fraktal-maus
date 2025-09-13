///// Otter: Renderer-Core – GL init + window, delegates to pipeline; no zoom logic here.
///// Schneefuchs: Strict CUDA/GL separation; deterministic ASCII logs; resources clearly owned.
///// Maus: Progressive cooldown + Tatze 7 soft-invalidate on view jumps (post-pipeline, no memset).
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

    // Create window + GL context
    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        LUCHS_LOG_HOST("[ERROR] Failed to create GLFW window");
        return false;
    }

    // createWindow() already makes context current; second call is harmless.
    glfwMakeContextCurrent(state.window);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGL context made current");
        if (glfwGetCurrentContext() != state.window) {
            LUCHS_LOG_HOST("[ERROR] Current OpenGL context is not the GLFW window!");
        } else {
            LUCHS_LOG_HOST("[CHECK] OpenGL context correctly bound to window");
        }
    }

    // GLEW init (idempotent enough; log errors clearly)
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        LUCHS_LOG_HOST("[FATAL] glewInit failed: %s", reinterpret_cast<const char*>(glewGetErrorString(glewErr)));
        RendererWindow::destroyWindow(state.window);
        state.window = nullptr;
        return false;
    }

    // Note: VSync preference is initialized exactly once in renderer_loop.cpp (initVSyncOnce).
    // Renderer core does not touch swap interval here to avoid conflicting policies.

    // Create GL resources: PBO + texture (immutable storage)
    if (!glResourcesInitialized) {
        OpenGLUtils::setGLResourceContext("init");
        for (auto& b : state.pboRing) { b = Hermelin::GLBuffer(OpenGLUtils::createPBO(state.width, state.height)); }
    state.pboIndex = 0;
        state.tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(state.width, state.height));

        // Register PBO with CUDA (CudaInterop encapsulates CUDA-13 compatible path)
        CudaInterop::registerPBO(state.currentPBO());

        // Clean GL state: unbind PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glResourcesInitialized = true;
        if constexpr (Settings::debugLogging) {
            GLint boundPBO = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPBO);
            LUCHS_LOG_HOST("[CHECK] initGL - GL PBO bound: %d | PBO ID: %u", boundPBO, state.currentPBO().id());
        }
    }

    glInitialized = true;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[INIT] GL init complete (w=%d h=%d)", state.width, state.height);
    }

    // Prepare GPU pipeline
    RendererPipeline::init();

    // Allocate device buffers (tileSize stays configured in pipeline/zoom logic)
    state.setupCudaBuffers(Settings::BASE_TILE_SIZE > 0 ? Settings::BASE_TILE_SIZE : 16);

    return true;
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::renderFrame() {
    // Delegate to pipeline: CUDA → analysis → upload → draw → logs
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] Entering Renderer::renderFrame");
    }

    // Progressive resume: set __constant__ program state once per frame (gated by cooldown)
    // Enables kernel-side "resume" only if: global + runtime flags true, state buffers present, and not in cooldown.
    {
        const bool inCooldown = (state.progressiveCooldownFrames > 0);
        const bool progReady =
            Settings::progressiveEnabled &&
            state.progressiveEnabled &&
            (state.d_stateZ.get()  != nullptr) &&
            (state.d_stateIt.get() != nullptr) &&
            !inCooldown;

        const int addIter = Settings::progressiveAddIter; // per-frame budget
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

    // Execute one frame (no device-wide sync in this path)
    FramePipeline::execute(state);

    // ---------- Tatze 7: Soft-invalidate on abrupt view change (post-pipeline) ----------
    // Detect large jumps in zoom or center (screen-space) and pause progressive resume for one frame.
    // Thresholds: zoomRatio >= 1.5 OR pan >= 8.0 px → soft invalidate (no memset).
    {
        struct PrevView { bool have=false; double zoom=0.0, cx=0.0, cy=0.0; };
        static PrevView prev;

        bool justInvalidated = false;

        if (prev.have) {
            const double z0 = std::max(prev.zoom, 1e-30);
            const double z1 = std::max((double)state.zoom, 1e-30);
            const double zoomRatio = (z1 > z0) ? (z1 / z0) : (z0 / z1);

            // Screen-space pan in pixels
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

        // Decrement cooldown unless we invalidated just now; when it hits zero, re-enable progressive.
        if (state.progressiveCooldownFrames > 0) {
            if (!justInvalidated) {
                --state.progressiveCooldownFrames;
                if (state.progressiveCooldownFrames == 0) {
                    state.progressiveEnabled = true;
                    if constexpr (Settings::debugLogging) {
                        LUCHS_LOG_HOST("[PROG] cooldown finished -> progressiveEnabled=true");
                    }
                }
            } else {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[PROG] cooldown kept at %d (fresh invalidate this frame)",
                                   state.progressiveCooldownFrames);
                }
            }
        }

        // Update previous view state at end of frame
        prev.zoom = (double)state.zoom;
        prev.cx   = (double)state.center.x;
        prev.cy   = (double)state.center.y;
        prev.have = true;
    }
    // ---------- End Tatze 7 --------------------------------------------------------

    // Present window / pump events
    glfwSwapBuffers(state.window);
    glfwPollEvents();
}

void Renderer::freeDeviceBuffers() {
    // GPU/GL buffers
    state.d_iterations.free();
    state.d_entropy.free();
    state.d_contrast.free();

    // Progressive per-pixel state
    state.d_stateZ.free();
    state.d_stateIt.free();

    for (auto& b : state.pboRing) { b.free(); }
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

    // State updates PBO/texture/device buffers and re-registers PBO if needed.
    state.resize(newW, newH);
}

void Renderer::cleanup() {
    if (!glInitialized) return;

    // GL-dependent: pipeline first
    RendererPipeline::cleanup();
    HeatmapOverlay::cleanup();

    // Release CUDA interop (no GL context required)
    CudaInterop::unregisterPBO();

    // Destroy window (and GL context)
    RendererWindow::destroyWindow(state.window);

    // Host-side
    freeDeviceBuffers();
    glfwTerminate();

    glInitialized = false;
    glResourcesInitialized = false;
}
