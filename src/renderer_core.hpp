// Datei: src/renderer_core.hpp
// Zeilen: 44
// 🐭 Maus-Kommentar: Header für das Rendering-Modul. initGL() meldet Erfolg/Fails. Internes Flag glInitialized schützt vor destruktivem Leichtsinn. Schneefuchs: „Kein Kontext, kein Cleanup.“

#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"

class Renderer {
public:
Renderer(int width, int height);
~Renderer();

bool initGL();  // 🟢 gibt Erfolg zurück
void renderFrame_impl(bool autoZoomEnabled);
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
