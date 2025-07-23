// Datei: src/hud.cpp
// Zeilen: 111
// 🐭 Maus-Kommentar: HUD-Textbox hat exakt dieselben Abstände wie Heatmap (oben/links = 16). Keine Koordinatenverwirrung mehr – alles Otter-symmetrisch!

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"
#pragma warning(disable:4505) // nötig für stb_easy_font.h wegen ungenutzter interner Funktionen (Otter geprüft)
#include <locale.h>
#include <cstdio>

namespace Hud {

static GLuint vao = 0, vbo = 0;

void draw(RendererState& state) {
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, (void*)0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, state.width, state.height, 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_TEXTURE_2D);

    // === Layout-Konstanten ===
    constexpr float margin = 16.0f;
    const float startX = margin;
    const float startY = margin;
    const float lineHeight = 40.0f;
    const float padding = 10.0f;

    const char* lines[] = {
        "HUD ACTIVE",
        nullptr,
        nullptr,
        nullptr
    };

    setlocale(LC_NUMERIC, "C");
    char buf[128];

    std::snprintf(buf, sizeof(buf), "FPS: %.0f", state.fps);
    lines[1] = _strdup(buf);

    double z = 1.0 / double(state.zoom);
    int exponent = int(std::log10(z));
    std::snprintf(buf, sizeof(buf), "Zoom: 1e%d", exponent);
    lines[2] = _strdup(buf);

    std::snprintf(buf, sizeof(buf), "Offset: %.6f, %.6f", state.offset.x, state.offset.y);
    lines[3] = _strdup(buf);

    // === Hintergrundbox ===
    float blockHeight = lineHeight * 4 + 2 * padding;
    float blockWidth = 400.0f;
    float bx = startX;
    float by = startY;

    float bg[] = {
        bx,           by,
        bx+blockWidth,by,
        bx+blockWidth,by+blockHeight,
        bx,           by+blockHeight
    };

    glColor4f(0.1f, 0.1f, 0.1f, 0.4f); // Hintergrund
    glBufferData(GL_ARRAY_BUFFER, sizeof(bg), bg, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_QUADS, 0, 4);

    glColor4f(0.3f, 0.3f, 0.3f, 0.7f); // Rahmen
    glBufferData(GL_ARRAY_BUFFER, sizeof(bg), bg, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINE_LOOP, 0, 4);

    // === Text zeichnen ===
    glColor3f(1, 1, 1);

    for (int i = 0; i < 4; ++i) {
        if (!lines[i] || lines[i][0] == '\0') continue;

        float buffer[9999]; // 🐭 Fix: vorher char[] → nun float[] wegen stb_easy_font (Otter geprüft)
        unsigned char color[4] = { 255, 255, 255, 255 };
        int quads = stb_easy_font_print(startX, startY + i * lineHeight, (char*)lines[i], color, buffer, sizeof(buffer));

        if (Settings::debugLogging)
            std::printf("[HUD] Line %d: '%s' -> %d quads\n", i, lines[i], quads);

        if (quads > 0) {
            glBufferData(GL_ARRAY_BUFFER, quads * 4 * sizeof(float) * 2, buffer, GL_DYNAMIC_DRAW);
            glDrawArrays(GL_QUADS, 0, quads * 4);

            if (Settings::debugLogging) {
                GLenum err = glGetError();
                if (err != GL_NO_ERROR)
                    std::printf("[HUD] OpenGL error after draw: 0x%x\n", err);
            }
        }
    }

    for (int i = 1; i < 4; ++i) free((void*)lines[i]);

    glDisableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_TEXTURE_2D);

    glMatrixMode(GL_MODELVIEW);   glPopMatrix();
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glBindVertexArray(0);
}

} // namespace Hud
