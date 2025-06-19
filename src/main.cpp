// Datei: src/main.cpp
// Zeilen: 20
// 🐭 Maus-Kommentar: Hauptprogramm – Einstiegspunkt. Jetzt mit aktiver Ressourcenkapselung über getState(). Schneefuchs meint: „Wer Zugriff will, soll darum bitten – höflich!“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"  // 🧠 Für initResources()

int main() {
    #if defined(DEBUG) || defined(_DEBUG)
        if (Settings::debugLogging) {
            std::puts("[DEBUG] Mandelbrot-Otterdream gestartet");
        }
    #endif

    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    // 🔧 Init von PBO, Textur, CUDA-Buffern, HUD über gekapselten Zugriff
    initResources(renderer.getState());

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true); // Auto-Zoom aktiviert
    }

    return 0;
}
