#pragma once

extern "C" {
int stb_easy_font_print(float x, float y, const char* text, const unsigned char* color_rgb, void* vertex_buffer, int vbuf_size);
}

namespace Hud {
void init();
void draw(float fps, float frameTimeMs, float zoom, float offX, float offY, int w, int h);
void cleanup();
}
