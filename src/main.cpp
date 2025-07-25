// Datei: src/main.cpp
// 🐭 Maus-Kommentar: Overlay wird direkt über RendererState initialisiert. Heatmap-Status aus Settings. Kompakt, klar, kein Overhead. Schneefuchs nickt.
#include "pch.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"

int main() {        
    if (Settings::debugLogging)
        std::puts("[DEBUG] Mandelbrot-Otterdream gestartet");

    Renderer renderer(Settings::width, Settings::height);
    if (!renderer.initGL()) {
        std::puts("[FATAL] OpenGL-Initialisierung fehlgeschlagen - Programm wird beendet");
        return EXIT_FAILURE;
    }

    RendererLoop::initResources(renderer.getState());
    renderer.getState().heatmapOverlayEnabled = Settings::heatmapOverlayEnabled;
    renderer.getState().warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;
    CudaInterop::setPauseZoom(false);

    while (!renderer.shouldClose()) {
        renderer.renderFrame_impl(); // Kein Parameter mehr (autoZoomEnabled entfernt)
        glfwPollEvents();
        glfwSwapBuffers(renderer.getState().window);
    }

    return 0;
}
