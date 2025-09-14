///// Otter: Main sets up renderer, CUDA precheck, and the tight frame loop; no GLFW error callback here (moved to RendererWindow after glfwInit).
///// Schneefuchs: Headers/sources in sync; CUDA_CHECK lives in luchs_log_host.hpp; no pre-init GLFW calls in main.
///// Maus: Precise timing; overlays follow Settings; deterministic ASCII logs; thread-safe logging.
///// Datei: src/main.cpp

#include "pch.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"
#include "luchs_log_host.hpp"

#include <chrono>
#include <cstdlib>
#include <GLFW/glfw3.h>

int main()
{
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[BOOT] Mandelbrot-Otterdream started");

    // NOTE: (2) GLFW init & error callback are handled inside RendererWindow::createWindow()
    //       after glfwInit(). Do NOT call any GLFW functions here before initGL().

    // Create renderer (GL context/window is created inside initGL by project convention)
    Renderer renderer(Settings::width, Settings::height);
    if (!renderer.initGL())
    {
        LUCHS_LOG_HOST("[FATAL] OpenGL initialization failed - aborting");
        return EXIT_FAILURE;
    }

    // CUDA early sanity (device presence etc.)
    if (!CudaInterop::precheckCudaRuntime())
    {
        LUCHS_LOG_HOST("[FATAL] CUDA pre-initialization failed - no usable device");
        return EXIT_FAILURE;
    }

    // Optional: verify cudaGetErrorString availability/path (project-specific guard)
    if (!CudaInterop::verifyCudaGetErrorStringSafe())
    {
        LUCHS_LOG_HOST("[FATAL] CUDA error-string path invalid - refusing to proceed");
        return EXIT_FAILURE;
    }

    // Sync overlay flags from Settings (runtime-visible defaults)
    renderer.getState().heatmapOverlayEnabled       = Settings::heatmapOverlayEnabled;
    renderer.getState().warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;

    // Ensure zoom is running unless paused elsewhere
    CudaInterop::setPauseZoom(false);

    // Defensive check: window pointer must be valid for swap/poll
    if (!renderer.getState().window)
    {
        LUCHS_LOG_HOST("[FATAL] RendererState.window is null - cannot run frame loop");
        return EXIT_FAILURE;
    }

    while (!renderer.shouldClose())
    {
        auto frameStart = std::chrono::high_resolution_clock::now();

        // Core frame: render everything through the loop
        RendererLoop::renderFrame_impl(renderer.getState());

        // Pump events before swap for lower input latency
        glfwPollEvents();

        auto swapStart = std::chrono::high_resolution_clock::now();
        glfwSwapBuffers(renderer.getState().window);
        auto swapEnd = std::chrono::high_resolution_clock::now();

        if (Settings::debugLogging)
        {
            const float swapMs  = std::chrono::duration<float, std::milli>(swapEnd - swapStart).count();
            const float totalMs = std::chrono::duration<float, std::milli>(swapEnd - frameStart).count();
            LUCHS_LOG_HOST("[FRAME] swap=%.2fms total=%.2fms", swapMs, totalMs);
        }
    }

    LUCHS_LOG_HOST("[EXIT] Clean shutdown");
    return EXIT_SUCCESS;
}
