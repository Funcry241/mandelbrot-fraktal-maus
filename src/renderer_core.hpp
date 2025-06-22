// Datei: src/renderer_core.hpp
// Zeilen: 41
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. `initGL()` liefert jetzt bool – für verlässliche Fehlererkennung. Schneefuchs: „Wer nicht antwortet, wird auch nicht gefragt.“

#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool initGL();  // 🟢 war void – jetzt bool für Fehlerprüfung
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
