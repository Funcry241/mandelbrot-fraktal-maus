// Datei: src/hud.hpp

#ifndef HUD_HPP
#define HUD_HPP

#include <string>

extern "C" {
    int stb_easy_font_print(float x, float y, const char *text,
                            const unsigned char *color_rgb,
                            void *vertex_buffer, int vbuf_size);
}

namespace Hud {

    /// Initialisiert HUD-Shader und -Puffer
    void init();

    /// Zeichnet den kompletten HUD (FPS + Zoom/Offset).
    void draw(float fps,
              float zoom,
              float offsetX,
              float offsetY,
              int width,
              int height);

    /// Gibt alle OpenGL-Ressourcen des HUD frei
    void cleanup();

}

#endif // HUD_HPP
