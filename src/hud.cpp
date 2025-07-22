// Datei: src/hud.cpp
// Zeilen: 79
// 🐭 Maus-Kommentar: Projekt Phönix – FreeType abgestreift, EasyFont erhoben. Kein Shader, keine Init. Zeichnet direkt per CPU-gepuffertem ASCII. Schnell, robust, restartfrei. Otter: „HUD ist jetzt ein Lichtschalter.“

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#pragma warning(disable: 4505) // unreferenzierte statische Funktionen
#include "stb_easy_font.h"

namespace Hud {

// 🧱 Einfacher Quad-Puffer
static GLuint vao = 0, vbo = 0;

void draw(RendererState& state) {
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);

    auto drawText = [](const char* text, float x, float y) {
        char buffer[9999]; // Max 999 chars
        int quads = stb_easy_font_print(x, y, (char*)text, nullptr, buffer, sizeof(buffer));

        glBufferData(GL_ARRAY_BUFFER, quads * 4 * sizeof(float) * 4, buffer, GL_DYNAMIC_DRAW);
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

    char buf[128];
    std::snprintf(buf, sizeof(buf), "FPS: %.1f", state.fps);
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
