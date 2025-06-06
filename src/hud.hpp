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

    /// Initializes HUD shaders and buffers
    void init();

    /// Draws the complete HUD (FPS, Frame Time, Zoom/Offset).
    void draw(float fps,
              float frameTimeMs,   // 🐭 NEU: Frame Time in ms
              float zoom,
              float offsetX,
              float offsetY,
              int width,
              int height);

    /// Frees all OpenGL resources of the HUD
    void cleanup();

}

#endif // HUD_HPP
