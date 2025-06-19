// Datei: src/renderer_core.hpp
// Zeilen: 39
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Zugriff auf `state` bleibt öffentlich für HUD & Loop. `renderFrame_impl` ist jetzt private, da intern genutzt. Überflüssige Methoden entfernt – Schneefuchs nickt mit strenger Miene.

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

    RendererState state;  // ⚠️ öffentlich, da z. B. von HUD verwendet

private:
    void renderFrame_impl(bool autoZoomEnabled);  // 🔐 nur intern aufrufbar
    void setupBuffers();
    void freeDeviceBuffers();
};
