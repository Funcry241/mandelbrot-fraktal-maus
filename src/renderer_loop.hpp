// Datei: src/renderer_loop.hpp
// Zeilen: 21
// 🐭 Maus-Kommentar: Definiert den Render-Loop und die Darstellung pro Frame. Steuert FPS-Zähler, Auto-Zoom-Logik und HUD-Anzeige. Schneefuchs würde sagen: „Der Taktgeber des Fraktal-Tanzes.“

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

} // namespace RendererLoop
