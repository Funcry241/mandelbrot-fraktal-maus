// Datei: src/renderer_loop.hpp
// Zeilen: 32
// 🐭 Maus-Kommentar: Definiert den Render-Loop und die Darstellung pro Frame. Steuert FPS-Zähler, Auto-Zoom-Logik und HUD-Anzeige. Schneefuchs würde sagen: „Der Taktgeber des Fraktal-Tanzes.“

#pragma once

// Nur GLFW ist nötig – keine CUDA oder GLEW im Header!
#include <GLFW/glfw3.h>

#include "renderer_state.hpp"
#include "common.hpp"

namespace RendererLoop {

// 🆕 Initialisiert OpenGL-, CUDA- und HUD-Ressourcen
void initResources(RendererState& state);

// 🕒 Initialisiert Zeitmesser, misst deltaTime & berechnet FPS
void beginFrame(RendererState& state);

// 🔢 Passt dynamisch die Tile-Größe an (z. B. bei Zoomänderung)
void updateTileSize(RendererState& state);

// ⚙️ Startet CUDA-Rendering, analysiert Entropie, berechnet neues Ziel
void computeCudaFrame(RendererState& state);

// 🎯 Aktualisiert Zoom & Offset per LERP, falls Auto-Zoom aktiv
void updateAutoZoom(RendererState& state);

// 🖼️ Zeichnet das Frame (Textur + HUD), tauscht OpenGL-Buffer
void drawFrame(RendererState& state);

// 🔁 Interner Ablauf: Alle Render-Schritte eines Frames
void renderFrame_impl(RendererState& state, bool autoZoomEnabled);

// 🚪 Externe API: Wird von `renderer_core.cu` oder `main.cpp` aufgerufen
void renderFrame(RendererState& state, bool autoZoomEnabled);

} // namespace RendererLoop
