// hud.cpp

#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

#include "hud.hpp"

#ifndef __CUDACC__
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#include <cstdio>
#include <cstring>

namespace Hud {

    void draw(float fps,
              float zoom,
              float offsetX,
              float offsetY,
              int width,
              int height)
    {
        // **1) Orthographische Projektion aktivieren**
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        // (0,0) oben links, (width,height) unten rechts
        glOrtho(0, width, height, 0, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // **2) Textfarbe weiß setzen**
        glColor3f(1.0f, 1.0f, 1.0f);

        // **3) STB_EASY_FONT Buffer** (genug Platz reservieren)
        static char vertex_buffer[10000];

        // **4a) FPS zeichnen (links oben, 10px Abstand)**
        char text_fps[64];
        std::snprintf(text_fps, sizeof(text_fps), "FPS: %.1f", fps);
        int quads_fps = stb_easy_font_print(
            10.0f, 10.0f,
            text_fps,
            nullptr,               // nullptr → weiße Schrift
            vertex_buffer,
            sizeof(vertex_buffer)
        );
        glEnableClientState(GL_VERTEX_ARRAY);
        // 2 floats pro Vertex, stride=16 (Standard für stb_easy_font)
        glVertexPointer(2, GL_FLOAT, 16, vertex_buffer);
        glDrawArrays(GL_QUADS, 0, quads_fps * 4);
        glDisableClientState(GL_VERTEX_ARRAY);

        // **4b) Zoom & Offset darunter (y = 30px)**
        char text_zoom[128];
        std::snprintf(
            text_zoom, sizeof(text_zoom),
            "Zoom: %.2f, Offset: (%.4f, %.4f)",
            zoom, offsetX, offsetY
        );
        int quads_zoom = stb_easy_font_print(
            10.0f, 30.0f,
            text_zoom,
            nullptr,
            vertex_buffer,
            sizeof(vertex_buffer)
        );
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 16, vertex_buffer);
        glDrawArrays(GL_QUADS, 0, quads_zoom * 4);
        glDisableClientState(GL_VERTEX_ARRAY);

        // **5) Projection/Modelview wieder restaurieren**
        glPopMatrix();                // Modelview zurücksetzen
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();                // Projection zurücksetzen
        glMatrixMode(GL_MODELVIEW);   // Für den Fall, dass der Aufrufer noch Modelview erwartet

        // **Optional**: Depth‐Test & Blending muss der Aufrufer selbst setzen, falls nötig.
    }

} // namespace Hud
