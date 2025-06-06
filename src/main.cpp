// Datei: src/main.cpp

#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"   // 🐭 Added to call checkDynamicParallelismSupport()

int main() {
    // 🐭 Check if the GPU supports dynamic parallelism before anything else
    CudaInterop::checkDynamicParallelismSupport();

    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    while (!renderer.shouldClose()) {
        renderer.renderFrame();
    }

    renderer.cleanup();
    return 0;
}
