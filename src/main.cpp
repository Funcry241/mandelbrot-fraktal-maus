// Datei: src/main.cpp
#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "progressive.hpp"  // 🐭 Jetzt richtig eingebunden für Iterations-Handling

int main() {
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    while (!renderer.shouldClose()) {
        renderer.renderFrame();
        incrementIterations();  // 🐭 Nach jedem Frame Iterationen erhöhen
    }

    return 0;
}
