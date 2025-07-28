// Datei: src/renderer_loop.hpp
// 🐭 Maus-Kommentar: Render-Loop, FPS, HUD, Auto-Zoom und Eingaben jetzt sauber gekapselt. Kein autoZoomEnabled-Parameter mehr – Signatur jetzt final synchron mit renderer_core. drawOverlay ist lokal. Schneefuchs: „Der Taktgeber des Fraktal-Tanzes, mit Blick für Tasten.“
#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"
#include "frame_context.hpp"

namespace RendererLoop {

// 🕒 Startet Frame: Zeitmesser, FPS, Delta berechnen
void beginFrame(RendererState& state);

// 🎬 Kompletter Frame: CUDA, Auto-Zoom, Textur, HUD, Overlay
void renderFrame_impl(RendererState& state);  // autoZoomEnabled entfernt!

// 🎹 Tastatur-Callback für GLFW (Overlay, Pause, Zoom etc.)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

} // namespace RendererLoop
