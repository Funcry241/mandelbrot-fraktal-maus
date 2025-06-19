// Datei: src/hud.hpp
// Zeilen: 44
// 🐭 Maus-Kommentar: HUD-Schnittstelle – zeigt FPS, Zoom und Offsets via STB-Easy-Font. Wird über OpenGL-Overlay gezeichnet. Kein ImGui, keine Abhängigkeiten, dafür pures ASCII mit 1ms Genauigkeit. Schneefuchs hätte es „effiziente Eleganz“ genannt.

#pragma once

#include <string>
#include "renderer_state.hpp"  // ✅ Damit RendererState in draw() bekannt ist

// ------------------------------------------------------------
// 🖥️ STB Easy Font Binding (nur für interne Nutzung)
// ------------------------------------------------------------
extern "C" {
    int stb_easy_font_print(float x, float y, const char* text,
                            const unsigned char* color_rgb,
                            void* vertex_buffer, int vbuf_size);
}

// ------------------------------------------------------------
// 🎯 HUD-Overlay: FPS, FrameTime, Zoom-Level, Offsets
// ------------------------------------------------------------
namespace Hud {

    /// 🚀 Initialisiert Shader und Vertex-Buffer für das HUD
    void init();

    /// 🖼️ Rendert das HUD mit den aktuellen Werten aus dem RendererState
    void draw(RendererState& state);  // ✅ Vereinheitlicht – Übergabe des gesamten Zustands

    /// 🧹 Gibt alle HUD-bezogenen OpenGL-Ressourcen frei
    void cleanup();

} // namespace Hud
