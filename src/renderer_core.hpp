// Datei: src/renderer_core.hpp
// Zeilen: 42
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. `state` ist jetzt geschützt, Zugriff über `getState()`. `cleanup()` bereit für sauberes Shutdown. Schneefuchs: „Zugriff ja – aber mit Stil.“

#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void initGL();
    void renderFrame(bool autoZoomEnabled);
    bool shouldClose() const;
    void resize(int newW, int newH);

    // 🆕 Getter für Zugriff auf internen Zustand
    RendererState& getState() { return state; }

private:
    RendererState state;  // 🔐 jetzt privat, aber via getState() zugänglich

    void renderFrame_impl(bool autoZoomEnabled);
    void setupBuffers();
    void freeDeviceBuffers();
    void cleanup();
};
