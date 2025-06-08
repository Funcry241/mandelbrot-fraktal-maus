// Datei: src/main.cpp
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
        int pState     = glfwGetKey(window, GLFW_KEY_P); // 🐭 Taste P zum Pausieren

        // 🐭 Toggle Auto-Zoom mit SPACE
        if (spaceState == GLFW_PRESS && !spaceWasPressed) {
            autoZoomEnabled = !autoZoomEnabled;
            spaceWasPressed = true;
            std::printf("[INFO] Auto-Zoom %s\n", autoZoomEnabled ? "ENABLED" : "DISABLED");
        }
        if (spaceState == GLFW_RELEASE) {
            spaceWasPressed = false;
        }

        // 🐭 Toggle Pause/Resume Zoom mit P
        if (pState == GLFW_PRESS && !pauseWasPressed) {
            bool currentPauseState = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!currentPauseState);
            pauseWasPressed = true;
            std::printf("[INFO] Zoom %s\n", !currentPauseState ? "PAUSED" : "RESUMED");
        }
        if (pState == GLFW_RELEASE) {
            pauseWasPressed = false;
        }

        renderer.renderFrame(autoZoomEnabled);

        // 🐭 Iterationen immer erhöhen, egal ob Zoom aktiv
        incrementIterations();
    }

    return 0;
}
