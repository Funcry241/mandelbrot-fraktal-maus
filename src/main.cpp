// Datei: src/main.cpp
// Zeilen: 22
// 🐭 Maus-Kommentar: Hauptprogramm – Einstiegspunkt. Jetzt mit korrekt gekapseltem Ressourcenzugriff über RendererLoop::initResources(). Schneefuchs nickt: „Niemals global, immer modular!“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"  // 🧠 Für RendererLoop::initResources

int main() {
    if (Settings::debugLogging) {
        std::puts("[DEBUG] Mandelbrot-Otterdream gestartet");
    }

    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    // 🔧 Init von PBO, Textur, CUDA-Buffern, HUD über modularisierte Schnittstelle
    RendererLoop::initResources(renderer.getState());

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true); // Auto-Zoom aktiviert
        glfwPollEvents();
    }

    return 0;
}
