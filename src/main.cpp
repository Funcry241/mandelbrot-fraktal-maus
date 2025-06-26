// Datei: src/main.cpp
// Zeilen: 32
// 🐭 Maus-Kommentar: Hauptprogramm – entfernt globalRendererState komplett. RendererState wird nun direkt weitergereicht. Schneefuchs: „Globale Zustände? Nur wenn du die Welt regierst.“

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

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true);
        glfwPollEvents();
    }

    return 0;
}
