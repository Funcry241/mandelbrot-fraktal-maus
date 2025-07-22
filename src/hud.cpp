// Datei: src/hud.cpp
// Zeilen: 91
// 🐭 Maus-Kommentar: Sichtbarkeit getestet mit "HUD ACTIVE". Logging ASCII-safe. Shader-State wird nun zurückgesetzt. Otter: „HUD kommt klar, wenn keiner reinpfuscht.“

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#pragma warning(disable: 4505)
#include "stb_easy_font.h"
#include <locale.h>

namespace Hud {

static GLuint vao = 0, vbo = 0;

void draw(RendererState& state) {
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    glUseProgram(0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    auto drawText = [](const char* text, float x, float y) {
        char buffer[9999];
        char local[256];
        strcpy_s(local, sizeof(local), text);

        int quads = stb_easy_font_print(x, y, local, nullptr, buffer, sizeof(buffer));

        if (Settings::debugLogging) {
            printf("[HUD] Drawing \"%s\" -> %d quads\n", local, quads);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, quads * 4 * sizeof(float) * 2, buffer, GL_DYNAMIC_DRAW);
        glDrawArrays(GL_QUADS, 0, quads * 4);
    };

    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, state.width, state.height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_TEXTURE_2D);
    glColor3f(1, 1, 1);

    drawText("HUD ACTIVE", 20, 20);

    setlocale(LC_NUMERIC, "C");

    char buf[128];
    std::snprintf(buf, sizeof(buf), "FPS: %.0f", state.fps);
    drawText(buf, 20, 40);

    double z = 1.0 / double(state.zoom);
    int exponent = int(std::log10(z));
    std::snprintf(buf, sizeof(buf), "Zoom: 1e%d", exponent);
    drawText(buf, 20, 80);

    std::snprintf(buf, sizeof(buf), "Offset: %.6f, %.6f", state.offset.x, state.offset.y);
    drawText(buf, 20, 120);

    glEnable(GL_TEXTURE_2D);
    glPopMatrix();
    glBindVertexArray(0);
}

} // namespace Hud
