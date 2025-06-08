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

    while (!renderer.shouldClose()) {
        // 🐭 Toggle Auto-Zoom mit Leertaste
        GLFWwindow* window = renderer.getWindow();
        int spaceState = glfwGetKey(window, GLFW_KEY_SPACE);

        if (spaceState == GLFW_PRESS && !spaceWasPressed) {
            autoZoomEnabled = !autoZoomEnabled;
            spaceWasPressed = true;
            std::printf("[INFO] Auto-Zoom %s\n", autoZoomEnabled ? "ENABLED" : "DISABLED");
        }
        if (spaceState == GLFW_RELEASE) {
            spaceWasPressed = false;
        }

        renderer.renderFrame(autoZoomEnabled);

        // 🐭 Iterationen immer erhöhen, egal ob Zoom aktiv
        incrementIterations();
    }

    return 0;
}
