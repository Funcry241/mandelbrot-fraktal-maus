// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Keine überflüssigen Parameter mehr - volle Synchronität zur Source. Funktioniert exakt mit der einparametrigen Loop-Signatur. Otter und Schneefuchs sind stolz.
// 🦦 Otter: Kontextlogik korrekt einkapsuliert - keine Frühregistrierung möglich.
// 🦊 Schneefuchs: Header synchron zur Source. Keine Schattenvariablen. Keine Lücken.

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
    bool glInitialized = false;           // Cleanup-Schutz
    bool glResourcesInitialized = false;  // Kontextschutz: PBO/Texture erst nach glfwMakeContextCurrent

    void freeDeviceBuffers();
    void cleanup();
};
