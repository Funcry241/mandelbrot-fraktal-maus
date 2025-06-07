// Datei: src/main.cpp
#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "progressive.hpp"

int main() {
    Renderer r(Settings::width, Settings::height);
    r.initGL();
    while (!r.shouldClose()) {
        r.renderFrame();
        incrementIterations();
    }
}
