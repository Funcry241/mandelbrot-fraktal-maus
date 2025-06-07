#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "progressive.hpp"
#include <iostream>  // <--- Wichtig!

int main() {
    try {
        std::cout << "[INFO] Creating renderer..." << std::endl;
        Renderer r(Settings::width, Settings::height);

        std::cout << "[INFO] Initializing OpenGL..." << std::endl;
        r.initGL();
        std::cout << "[INFO] OpenGL Initialized." << std::endl;

        while (!r.shouldClose()) {
            r.renderFrame();
            incrementIterations();
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception caught!" << std::endl;
        return EXIT_FAILURE;
    }
}
