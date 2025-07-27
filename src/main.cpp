// Datei: src/main.cpp
// 🐭 Maus-Kommentar: Main-Loop ruft nur noch Renderer::renderFrame ohne Parameter. Setup ist klar, Logging vollständig integriert. Schneefuchs: „Eleganter geht’s nicht.“

#include "pch.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"
#include "luchs_log_host.hpp"
#include <chrono>

int main() {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] Mandelbrot-Otterdream gestartet");

    Renderer renderer(Settings::width, Settings::height);
    if (!renderer.initGL()) {
        LUCHS_LOG_HOST("[FATAL] OpenGL-Initialisierung fehlgeschlagen – Programm wird beendet");
        return EXIT_FAILURE;
    }

    RendererLoop::initResources(renderer.getState());
    renderer.getState().heatmapOverlayEnabled = Settings::heatmapOverlayEnabled;
    renderer.getState().warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;
    CudaInterop::setPauseZoom(false);

    while (!renderer.shouldClose()) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        RendererLoop::renderFrame_impl(renderer.getState());
        glfwPollEvents();

        auto swapStart = std::chrono::high_resolution_clock::now();
        glfwSwapBuffers(renderer.getState().window);
        auto swapEnd = std::chrono::high_resolution_clock::now();

        if (Settings::debugLogging) {
            float swapMs  = std::chrono::duration<float, std::milli>(swapEnd - swapStart).count();
            float totalMs = std::chrono::duration<float, std::milli>(swapEnd - frameStart).count();
            LUCHS_LOG_HOST("[Frame] swap=%.2fms total=%.2fms", swapMs, totalMs);
        }
    }

    return 0;
}
