// Datei: src/main.cpp
// 🐭 Maus-Kommentar: Hauptschleife für OpenGL + CUDA Mandelbrot-Renderer mit Auto-Zoom & Tastenevents

#define GLEW_STATIC
#include <GL/glew.h>       // GLEW zuerst initialisieren
#include <GLFW/glfw3.h>    // danach GLFW

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

    std::printf("[INIT] Renderer initialized – entering main loop\n");

    while (!renderer.shouldClose()) {
        GLFWwindow* window = renderer.getWindow();

        // ⌨️ Tastenzustände lesen
        int spaceState = glfwGetKey(window, GLFW_KEY_SPACE);
        int pState     = glfwGetKey(window, GLFW_KEY_P); // 🐭 Taste P für Pause/Resume

        // 🐾 Auto-Zoom toggeln (Space)
        if (spaceState == GLFW_PRESS && !spaceWasPressed) {
            autoZoomEnabled = !autoZoomEnabled;
            spaceWasPressed = true;
            std::printf("[INPUT] Auto-Zoom %s\n", autoZoomEnabled ? "ENABLED" : "DISABLED");
        }
        if (spaceState == GLFW_RELEASE) {
            spaceWasPressed = false;
        }

        // 🐾 Pause/Resume toggeln (P)
        if (pState == GLFW_PRESS && !pauseWasPressed) {
            bool currentPauseState = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!currentPauseState);
            pauseWasPressed = true;
            std::printf("[INPUT] Zoom %s\n", !currentPauseState ? "PAUSED" : "RESUMED");
        }
        if (pState == GLFW_RELEASE) {
            pauseWasPressed = false;
        }

        // 🖼️ CUDA + OpenGL Frame rendern
        renderer.renderFrame(autoZoomEnabled);

        // 🧠 Iterationstiefe dynamisch erhöhen
        Progressive::incrementIterations();

        // 📤 Fenster aktualisieren (Swap Buffer)
        glfwSwapBuffers(window);       // 💡 Ohne das bleibt das Bild weiß!

        // 🕹️ Ereignisse verarbeiten (Tastatur, Maus, etc.)
        glfwPollEvents();              // 💡 Ohne das wird ESC & Close-Button ignoriert
    }

    std::printf("[SHUTDOWN] Application exited cleanly.\n");
    return 0;
}
