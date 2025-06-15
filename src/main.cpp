// Datei: src/main.cpp
// 🐭 Maus-Kommentar: Startpunkt des OtterDream-Renderers – steuert Hauptschleife, Tastenevents und Auto-Zoom

#define GLEW_STATIC
#include <GL/glew.h>        // GLEW zuerst, wegen OpenGL Extensions
#include <GLFW/glfw3.h>     // dann GLFW für Fenster + Eingabe

#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "progressive.hpp"

int main() {
    // 🎬 Renderer initialisieren mit Fenstergröße aus Settings
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    // 🔗 Registrierung des KeyCallbacks für direkte Reaktion auf Tastenereignisse
    glfwSetKeyCallback(renderer.getWindow(), CudaInterop::keyCallback);

    bool autoZoomEnabled = true;     // 🔍 Auto-Zoom aktiv?
    bool spaceWasPressed = false;    // ⌨️ Space-Debounce
    bool pauseWasPressed = false;    // ⌨️ P-Debounce

    std::puts("[INIT] Renderer initialized – entering main loop");

    // 🌀 Haupt-Renderloop: läuft bis Fenster geschlossen wird
    while (!renderer.shouldClose()) {
        GLFWwindow* window = renderer.getWindow();

        // ⌨️ Eingabe abfragen
        int spaceState = glfwGetKey(window, GLFW_KEY_SPACE);  // Space → Auto-Zoom an/aus
        int pState     = glfwGetKey(window, GLFW_KEY_P);      // P → Pause/Resume

        // 🔁 Space: Auto-Zoom toggeln
        if (spaceState == GLFW_PRESS && !spaceWasPressed) {
            autoZoomEnabled = !autoZoomEnabled;
            spaceWasPressed = true;
            std::printf("[INPUT] Auto-Zoom %s\n", autoZoomEnabled ? "ENABLED" : "DISABLED");
        }
        if (spaceState == GLFW_RELEASE) spaceWasPressed = false;

        // 🔁 P: Pause-Status toggeln
        if (pState == GLFW_PRESS && !pauseWasPressed) {
            bool isPaused = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!isPaused);
            pauseWasPressed = true;
            std::printf("[INPUT] Zoom %s\n", isPaused ? "RESUMED" : "PAUSED");
        }
        if (pState == GLFW_RELEASE) pauseWasPressed = false;

        // 🎨 Frame rendern (CUDA → OpenGL)
        renderer.renderFrame(autoZoomEnabled);

        // ⏫ Iterationen langsam steigern (Detailschärfe wächst)
        Progressive::incrementIterations();

        // ⚠️ Kein glfwSwapBuffers / glfwPollEvents hier!
        // Diese Aufrufe passieren intern in renderer.renderFrame()
    }

    std::puts("[SHUTDOWN] Application exited cleanly.");
    return 0;
}
