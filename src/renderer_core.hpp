///// Otter: Rendering-Modul (Header) – Kontext sauber gekapselt; keine Frühregistrierung.
///// Schneefuchs: Header/Source synchron; keine Schattenvariablen; deterministische API.
///// Maus: Keine überflüssigen Parameter; exakt passend zur Implementation.
///// Datei: src/renderer_core.hpp

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
