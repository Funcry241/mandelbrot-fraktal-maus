// Datei: src/renderer_core.hpp
// Zeilen: 43
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Keine überflüssigen Parameter mehr – volle Synchronität zur Source. Funktioniert exakt mit der einparametrigen Loop-Signatur. Otter und Schneefuchs sind stolz.
#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool initGL();           // 🟢 gibt Erfolg zurück
    void renderFrame_impl(); // Korrekt: KEIN Parameter!
    bool shouldClose() const;
    void resize(int newW, int newH);

    // Getter für den internen Zustand
    RendererState& getState()             { return state; }
    const RendererState& getState() const { return state; }

private:
    RendererState state;
    bool glInitialized = false; // Cleanup-Schutz

    void freeDeviceBuffers();
    void cleanup();
};
