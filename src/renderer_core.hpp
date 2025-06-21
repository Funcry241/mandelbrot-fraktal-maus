// Datei: src/renderer_core.hpp
// Zeilen: 41
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Entfernt: `setupBuffers()`. Neu: `const getState()` für sauberen lesenden Zugriff. Schneefuchs: „Nur wer gibt, darf auch nehmen – aber bitte ohne Schreibzugriff.“

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
    const RendererState& getState() const { return state; }  // 🆕 nur lesend

private:
    RendererState state;  // 🔐 jetzt privat, aber via getState() zugänglich

    void renderFrame_impl(bool autoZoomEnabled);
    void freeDeviceBuffers();
    void cleanup();
};
