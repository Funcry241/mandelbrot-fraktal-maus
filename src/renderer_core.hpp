// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Keine überflüssigen Parameter mehr - volle Synchronität zur Source. Otter und Schneefuchs sind stolz.
// 🦦 Otter: Kontextlogik korrekt einkapsuliert - keine Frühregistrierung möglich.
// 🦊 Schneefuchs: Header synchron zur Source. Keine Schattenvariablen. Keine Lücken.

#pragma once

#include "renderer_state.hpp"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    [[nodiscard]] bool initGL();
    [[nodiscard]] bool shouldClose() const;
    void resize(int newW, int newH);

    void renderFrame();

    [[nodiscard]] RendererState&       getState()       noexcept { return state; }
    [[nodiscard]] const RendererState& getState() const noexcept { return state; }

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

private:
    RendererState state;
    bool glInitialized = false;
    bool glResourcesInitialized = false;

    void freeDeviceBuffers();
    void cleanup();
};
