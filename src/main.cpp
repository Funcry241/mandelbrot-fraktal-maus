// Datei: src/main.cpp
// Zeilen: 34
// 🐭 Maus-Kommentar: Hauptprogramm – überprüft jetzt korrekt das Ergebnis von `initGL()`. Schneefuchs bestand darauf: „Wer blind initialisiert, stirbt auch blind.“

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

    if (!renderer.initGL()) {
        std::puts("[FATAL] OpenGL-Initialisierung fehlgeschlagen – Programm wird beendet");
        return EXIT_FAILURE;
    }

    RendererLoop::initResources(*CudaInterop::globalRendererState);

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true);
        glfwPollEvents();
    }

    return 0;
}
