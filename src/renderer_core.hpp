// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Keine überflüssigen Parameter mehr - volle Synchronität zur Source. Otter und Schneefuchs sind stolz.
// 🦦 Otter: Kontextlogik korrekt einkapsuliert - keine Frühregistrierung möglich.
// 🦊 Schneefuchs: Header synchron zur Source. Keine Schattenvariablen. Keine Lücken.

#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool initGL();
    bool shouldClose() const;
    void resize(int newW, int newH);

    void renderFrame();  // <-- 🧩 HINZUFÜGEN

    RendererState& getState()             { return state; }
    const RendererState& getState() const { return state; }

private:
    RendererState state;
    bool glInitialized = false;
    bool glResourcesInitialized = false;

    void freeDeviceBuffers();
    void cleanup();
};
