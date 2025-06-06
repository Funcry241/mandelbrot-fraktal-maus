// Datei: src/main.cpp
#include "renderer_core.hpp"
#include "settings.hpp"    // 🐭 Hinzufügen!

int main() {
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    while (!renderer.shouldClose()) {
        renderer.renderFrame();
    }

    renderer.cleanup();
    return 0;
}
