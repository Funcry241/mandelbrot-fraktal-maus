// Datei: src/main.cpp
// Zeilen: 32
// 🐭 Maus-Kommentar: Hauptprogramm – Initialisiert Renderer, verknüpft globalRendererState im Namespace CudaInterop korrekt und startet den Renderloop. Schneefuchs bestand darauf, dass globale Zustände sauber im Namensraum leben – nicht anonym herumschwirren wie Otter ohne Teich.

#include "pch.hpp"

#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"  // ❗️WICHTIG: Damit der Namespace bekannt ist

// ✅ Globale Referenz innerhalb des korrekten Namensraums definieren
namespace CudaInterop {
    RendererState* globalRendererState = nullptr;
}

int main() {
    if (Settings::debugLogging) {
        std::puts("[DEBUG] Mandelbrot-Otterdream gestartet");
    }

    Renderer renderer(Settings::width, Settings::height);
    CudaInterop::globalRendererState = &renderer.getState();

    renderer.initGL();
    RendererLoop::initResources(*CudaInterop::globalRendererState);

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true);
        glfwPollEvents();
    }

    return 0;
}
