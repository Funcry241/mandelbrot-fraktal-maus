// Datei: src/main.cpp
// Zeilen: 17
// 🐭 Maus-Kommentar: Hauptprogramm – Einstiegspunkt. Init jetzt vollständig: OpenGL, CUDA-Interop und HUD. Fraktale wachsen auf stabilem Boden. Schneefuchs: „Jetzt erst fliegt der Otter!“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"  // 🧠 Neu: für initResources()

int main() {
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    // 🔧 Init von PBO, Textur, CUDA-Buffern, HUD
    // RendererLoop::initResources(renderer.state);

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true); // Auto-Zoom aktiviert
    }

    return 0;
}

