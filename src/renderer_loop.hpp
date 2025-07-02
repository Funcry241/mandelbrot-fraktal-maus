// Datei: src/renderer_loop.hpp
// Zeilen: 23
// 🐭 Maus-Kommentar: Definiert den Render-Loop und die Darstellung pro Frame. Steuert FPS-Zähler, Auto-Zoom-Logik, HUD und Eingabe. Schneefuchs sagt: „Der Taktgeber des Fraktal-Tanzes mit Blick für Tasten.“

#pragma once

#include <GLFW/glfw3.h>
#include "renderer_state.hpp"

namespace RendererLoop {

// 🔧 Initialisiert OpenGL-, CUDA- und HUD-Ressourcen
void initResources(RendererState& state);

// 🕒 Initialisiert Zeitmesser, misst deltaTime & berechnet FPS
void beginFrame(RendererState& state);

// 🎬 Führt einen vollständigen Frame-Durchlauf aus (CUDA, AutoZoom, Textur, HUD)
void renderFrame_impl(RendererState& state, bool autoZoomEnabled);

// 🎹 Tastatur-Callback für GLFW (Heatmap Toggle, Zoom Pause etc.)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

} // namespace RendererLoop
