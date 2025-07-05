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

bool initGL();  // 🟢 war void – jetzt bool für Fehlerprüfung
void renderFrame_impl(bool autoZoomEnabled);
bool shouldClose() const;
void resize(int newW, int newH);

// 🆕 Getter für Zugriff auf internen Zustand
RendererState& getState() { return state; }
const RendererState& getState() const { return state; }  // 🆕 nur lesend    

private:
RendererState state; // 🔐 Zugriff nur über getState()
bool glInitialized = false; // 🆕 Cleanup-Schutz für Destruktor

void freeDeviceBuffers();
void cleanup();

};
