// Datei: src/hud.cpp
// Zeilen: 118
// 🐭 Maus-Kommentar: HUD-Fehleranalyse aktiv. Sichtbarkeit per rotem Rechteck, Blend-Setup, Matrix-Logs. Otter: „Knallhart sichtbar. Entweder rot, oder tot.“

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#pragma warning(disable: 4505)
#include "stb_easy_font.h"
#include <locale.h>

namespace Hud {

static GLuint vao = 0, vbo = 0;

void draw(RendererState& state) {
    printf("[HUD] draw() BEGIN – w=%d h=%d fps=%.1f zoom=%.4f\n", state.width, state.height, state.fps, state.zoom);

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
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, state.width, state.height, 0.0, -1.0, 1.0);

    GLfloat proj[16];
    glGetFloatv(GL_PROJECTION_MATRIX, proj);
    printf("[HUD] PROJ[0]=%.2f PROJ[5]=%.2f PROJ[10]=%.2f\n", proj[0], proj[5], proj[10]);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_TEXTURE_2D);
    glColor3f(1, 1, 1);

    // Sichtbarkeitsrechteck – rot
    glBegin(GL_QUADS);
    glColor3f(1, 0, 0);
    glVertex2f(10, 10);
    glVertex2f(110, 10);
    glVertex2f(110, 110);
    glVertex2f(10, 110);
    glEnd();

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

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glBindVertexArray(0);
}

} // namespace Hud
