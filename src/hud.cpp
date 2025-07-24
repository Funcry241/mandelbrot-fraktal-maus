// Datei: src/hud.cpp
// Zeilen: 111
// 🐭 Maus-Kommentar: HUD-Textbox hat exakt dieselben Abstände wie Heatmap (oben/links = 16). Schriftdarstellung nutzt jetzt originalgetreues stb_easy_font-Setup (static Buffer, stride 16, kein VBO). Speicherfehler mit geteiltem Buffer behoben – alle Strings haben jetzt eigene Quelle (Otter-proof).

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"
#pragma warning(disable:4505)
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

    // 🐭 vorher: ein Buffer überschrieben → Schrift unsichtbar
    // jetzt: jeder Text hat eigene temporäre Quelle
    char buf1[64], buf2[64], buf3[64];

    std::snprintf(buf1, sizeof(buf1), "FPS: %.0f", state.fps);
    lines[1] = _strdup(buf1);

    double z = 1.0 / double(state.zoom);
    int exponent = int(std::log10(z));
    std::snprintf(buf2, sizeof(buf2), "Zoom: 1e%d", exponent);
    lines[2] = _strdup(buf2);

    std::snprintf(buf3, sizeof(buf3), "Offset: %.6f, %.6f", state.offset.x, state.offset.y);
    lines[3] = _strdup(buf3);

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

    glColor4f(0.1f, 0.1f, 0.1f, 0.4f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bg), bg, GL_DYNAMIC_DRAW);
    glVertexPointer(2, GL_FLOAT, 0, (void*)0);
    glDrawArrays(GL_QUADS, 0, 4);

    glColor4f(0.3f, 0.3f, 0.3f, 0.7f);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bg), bg, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINE_LOOP, 0, 4);

    // === Text zeichnen ===
    static char fontBuffer[99999];

    for (int i = 0; i < 4; ++i) {
        if (!lines[i] || lines[i][0] == '\0') continue;

        int quads = stb_easy_font_print(
            startX,
            startY + i * lineHeight,
            (char*)lines[i],
            NULL,
            fontBuffer,
            sizeof(fontBuffer)
        );

        if (Settings::debugLogging)
            std::printf("[HUD] Line %d: '%s' -> %d quads\n", i, lines[i], quads);

        if (quads > 0) {
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertexPointer(2, GL_FLOAT, 16, fontBuffer);
            glDrawArrays(GL_QUADS, 0, quads * 4);
        }
    }

    for (int i = 1; i < 4; ++i)
        free((void*)lines[i]);

    glDisableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_TEXTURE_2D);

    glMatrixMode(GL_MODELVIEW);   glPopMatrix();
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glBindVertexArray(0);
}

} // namespace Hud
