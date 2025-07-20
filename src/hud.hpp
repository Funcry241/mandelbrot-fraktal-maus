// Datei: src/hud.hpp
// Zeilen: 31
// 🐭 Maus-Kommentar: HUD-Schnittstelle – rendert FPS, Zoom und Offset mit FreeType auf ein eigenes Overlay. Keine ASCII-Notlösung mehr. Glatte Linien, echte Typografie. Flight-Ready, Schneefuchs-approved.

#pragma once

#include <string>
#include "renderer_state.hpp" // ✅ Damit RendererState in draw() bekannt ist

namespace Hud {

// 🚀 Initialisiert FreeType, lädt Font und erzeugt Shader + Texture-Atlas
void init();

// 🖼️ Rendert das HUD (FPS, Zoom, Offsets) auf das OpenGL-Overlay
void draw(RendererState& state);

// 🧹 Gibt alle OpenGL- und FreeType-Ressourcen des HUD frei
void cleanup();

} // namespace Hud
