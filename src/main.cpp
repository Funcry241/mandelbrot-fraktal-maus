// Datei: src/main.cpp
// Zeilen: 34
// 🐭 Maus-Kommentar: Hauptprogramm – entfernt globalRendererState komplett. RendererState wird nun direkt weitergereicht. Schneefuchs: „Globale Zustände? Nur wenn du die Welt regierst.“
// 🔄 Auto-Zoom wird nun beim Start explizit aktiviert – kein manuelles SPACE/P nötig.

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

    // ⏯️ Auto-Zoom explizit aktivieren beim Start
    CudaInterop::setPauseZoom(false);

    while (!renderer.shouldClose()) {
        renderer.renderFrame_impl(true);
        glfwPollEvents();
        glfwSwapBuffers(renderer.getState().window); 
    }

    return 0;
}
