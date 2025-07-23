// Datei: src/hud.cpp
// Zeilen: 116
// 🐭 Maus-Kommentar: Alpha 50a – HUD-Sichtbarkeit erzwingt glViewport-Logik, Font-Fallback und Lokalisierungssicherheit. Alles ASCII. Otter vermutet: setlocale war’s.

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#pragma warning(disable: 4505)
#include "stb_easy_font.h"

namespace Hud {

static GLuint vao = 0, vbo = 0;

void draw(RendererState& state) {
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    // Log aktiven Viewport zur Verifikation
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    if (Settings::debugLogging) {
        printf("[HUD] Viewport: x=%d y=%d w=%d h=%d\n", viewport[0], viewport[1], viewport[2], viewport[3]);
    }

    glUseProgram(0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // Korrekte Orthoprojektion
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, state.width, state.height, 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_TEXTURE_2D);
    glColor3f(1, 1, 1);

    auto drawText = [](const char* text, float x, float y) {
        char buffer[9999];
        int quads = stb_easy_font_print(x, y, const_cast<char*>(text), nullptr, buffer, sizeof(buffer));

        if (Settings::debugLogging) {
            printf("[HUD] Drawing \"%s\" -> %d quads\n", text, quads);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, quads * 4 * sizeof(float) * 2, buffer, GL_DYNAMIC_DRAW);
        glDrawArrays(GL_QUADS, 0, quads * 4);
    };

    drawText("HUD ACTIVE", 20, 20);

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
