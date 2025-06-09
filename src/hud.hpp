// Datei: src/hud.hpp

#pragma once
#ifndef HUD_HPP
#define HUD_HPP

#include <string>

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

    /// 🖼️ Rendert das HUD mit den aktuellen Werten (FPS, Frame Time, Zoom, Offset)
    void draw(float fps,
              float frameTimeMs,  ///< Framezeit in Millisekunden
              float zoom,
              float offsetX,
              float offsetY,
              int width,
              int height);

    /// 🧹 Gibt alle HUD-bezogenen OpenGL-Ressourcen frei
    void cleanup();

} // namespace Hud

#endif // HUD_HPP
