// Datei: src/hud.cpp
// Zeilen: 110
// 🐭 Maus-Kommentar: Das HUD nutzt nun ein einheitliches, halbtransparentes Rechteck zur Gruppierung aller Textzeilen. Warnung 4505 explizit deaktiviert, da `stb_easy_font.h` interne Funktionen enthält, die nie referenziert werden – Otter-Setup bleibt `/WX`-kompatibel.

#include "pch.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "stb_easy_font.h"

// Warnung C4505: "nicht referenzierte Funktion mit interner Verknüpfung wurde entfernt"
// wird von stb_easy_font.h erzeugt, ist aber harmlos.
// Deaktivieren notwendig, damit /WX nicht greift.
#pragma warning(disable:4505) // nötig für stb_easy_font.h wegen ungenutzter interner Funktionen (Otter geprüft)

#include <locale.h>

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
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, state.width, state.height, 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_TEXTURE_2D);

    const float x = 20.0f;
    const float y0 = 20.0f;
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

    // === Hintergrundbox für alle Zeilen ===
    float blockHeight = lineHeight * 4 + 2 * padding;
    float blockWidth = 400.0f;
    float bx = x - padding;
    float by = y0 - padding;

    float bg[] = {
        bx,            by,
        bx+blockWidth, by,
        bx+blockWidth, by+blockHeight,
        bx,            by+blockHeight
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
        if (!lines[i]) continue;
        char buffer[9999];
        unsigned char dummyColor[4] = { 255, 255, 255, 255 };
        int quads = stb_easy_font_print(x, y0 + i * lineHeight, (char*)lines[i], dummyColor, buffer, sizeof(buffer));
        if (quads > 0) {
            glBufferData(GL_ARRAY_BUFFER, quads * 4 * sizeof(float) * 2, buffer, GL_DYNAMIC_DRAW);
            glDrawArrays(GL_QUADS, 0, quads * 4);
        }
    }

    for (int i = 1; i < 4; ++i) free((void*)lines[i]);

    glEnable(GL_TEXTURE_2D);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glBindVertexArray(0);
}

} // namespace Hud
