// Datei: src/main.cpp
// 🐭 Maus-Kommentar: Startpunkt des OtterDream-Renderers – steuert Hauptschleife und delegiert Tasteneingaben über Callback

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

    // ⌨️ Tasteneingaben über Callback verarbeiten (Space = Auto-Zoom, P = Pause)
    glfwSetKeyCallback(renderer.getWindow(), CudaInterop::keyCallback);

    std::puts("[INIT] Renderer initialized – entering main loop");

    // 🌀 Haupt-Renderloop: läuft bis Fenster geschlossen wird
    while (!renderer.shouldClose()) {
        // 🎨 Frame rendern (CUDA → OpenGL)
        renderer.renderFrame(/*autoZoomEnabled=*/true);

        // ⏫ Iterationen langsam steigern (Detailschärfe wächst)
        Progressive::incrementIterations();

        // ⚠️ Kein glfwSwapBuffers / glfwPollEvents hier!
        // Diese Aufrufe passieren intern in renderer.renderFrame()
    }

    std::puts("[SHUTDOWN] Application exited cleanly.");
    return 0;
}
