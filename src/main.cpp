// Zeilen: 35
// Datei: src/main.cpp
// 🐭 Maus-Kommentar: `HeatmapOverlay::setEnabled(...)` ist gestrichen – Overlay wird direkt über RendererState gesteuert. Wir initialisieren den Overlay-Zustand korrekt aus settings.hpp. Schneefuchs: „Was sichtbar ist, beginnt im State.“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"

int main() {
    if (Settings::debugLogging) {
        std::puts("[DEBUG] Mandelbrot-Otterdream gestartet");
    }

    Renderer renderer(Settings::width, Settings::height);

    if (!renderer.initGL()) {
        std::puts("[FATAL] OpenGL-Initialisierung fehlgeschlagen – Programm wird beendet");
        return EXIT_FAILURE;
    }

    RendererLoop::initResources(renderer.getState());

    // 🟢 Heatmap-Overlay: Initialzustand aus Settings übernehmen
    renderer.getState().overlayEnabled = Settings::heatmapOverlayEnabled;

    // ⏯️ Auto-Zoom explizit aktivieren beim Start
    CudaInterop::setPauseZoom(false);

    while (!renderer.shouldClose()) {
        renderer.renderFrame_impl(true);
        glfwPollEvents();
        glfwSwapBuffers(renderer.getState().window); 
    }

    return 0;
}
