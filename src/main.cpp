// Datei: src/main.cpp

#define GLEW_STATIC
#include <GL/glew.h>   // 🐭 GANZ WICHTIG: Erst GLEW!
#include <GLFW/glfw3.h> // Danach GLF

#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "progressive.hpp"

int main() {
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    bool autoZoomEnabled = true;
    bool spaceWasPressed = false;
    bool pauseWasPressed = false;

    while (!renderer.shouldClose()) {
        GLFWwindow* window = renderer.getWindow();

        int spaceState = glfwGetKey(window, GLFW_KEY_SPACE);
        int pState     = glfwGetKey(window, GLFW_KEY_P); // 🐭 Taste P für Pause/Resume

        // 🐾 Auto-Zoom toggeln (Space)
        if (spaceState == GLFW_PRESS && !spaceWasPressed) {
            autoZoomEnabled = !autoZoomEnabled;
            spaceWasPressed = true;
            std::printf("[INFO] Auto-Zoom %s\n", autoZoomEnabled ? "ENABLED" : "DISABLED");
        }
        if (spaceState == GLFW_RELEASE) {
            spaceWasPressed = false;
        }

        // 🐾 Pause/Resume toggeln (P)
        if (pState == GLFW_PRESS && !pauseWasPressed) {
            bool currentPauseState = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!currentPauseState);
            pauseWasPressed = true;
            std::printf("[INFO] Zoom %s\n", !currentPauseState ? "PAUSED" : "RESUMED");
        }
        if (pState == GLFW_RELEASE) {
            pauseWasPressed = false;
        }

        // 🖼️ Render Frame mit (de)aktiviertem Auto-Zoom
        renderer.renderFrame(autoZoomEnabled);

        // 🧠 Iterationstiefe progressiv erhöhen
        Progressive::incrementIterations(); // <--- Fix hier!
    }

    return 0;
}
