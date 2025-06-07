// Datei: src/main.cpp
#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"

int main() {
    // 🐭 Dynamic Parallelism Check entfernt – unnötig
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    while (!renderer.shouldClose())
        renderer.renderFrame();

    // 🐭 Kein explizites cleanup mehr nötig – wird im Renderer-Destruktor automatisch erledigt
    return 0;
}
