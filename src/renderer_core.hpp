// Datei: src/renderer_core.hpp
// Zeilen: 40
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. Zugriff auf `state` bleibt öffentlich für HUD & Loop. `cleanup()` ergänzt zur vollständigen Ressourcenfreigabe. Schneefuchs: „Wer zerstört, muss vorher deklarieren!“

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
    void cleanup();  // 🧹 vollständiges Aufräumen aller GL/CUDA-Ressourcen
};
