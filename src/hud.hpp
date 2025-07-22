// Datei: src/hud.hpp
// Zeilen: 27
// 🐭 Maus-Kommentar: Projekt Phönix – FreeType verbrannt, EasyFont auferstanden. Kein Shader, kein Init-Fragilität. ASCII sicher, ultraleicht. Schneefuchs nickt, Otter ruft: „Endlich!“

#pragma once

#include <string>
#include "renderer_state.hpp" // ✅ Damit RendererState in draw() bekannt ist

namespace Hud {

// 🛫 Kein Init mehr nötig – EasyFont braucht keine Vorinitialisierung
inline void init() {} // 🔥 Projekt Phönix: Dummy-Funktion

// 🖼️ Rendert das HUD (FPS, Zoom, Offsets) per stb_easy_font
void draw(RendererState& state);

// 🧹 Keine Cleanup-Ressourcen mehr nötig – EasyFont ist stateless
inline void cleanup() {} // 🔥 Projekt Phönix: Dummy-Funktion

} // namespace Hud
