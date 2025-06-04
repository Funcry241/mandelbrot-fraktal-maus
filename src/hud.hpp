// Datei: src/hud.hpp

#ifndef HUD_HPP
#define HUD_HPP

// ------------------------------------------------------------------------------------------------
// HUD.hpp
// Zeichnet in der OpenGL‐Fixed‐Function‐Pipeline:
//  - aktuelle FPS (oben links, 10px Abstand)
//  - Zoom‐Faktor + Offset (darunter)
// Voraussetzung: GL‐Kontext ist aktiv, Projection/Modelview in Default‐Zustand
// ------------------------------------------------------------------------------------------------

#include <string>

// Forward‐Deklaration von stb_easy_font_print:
extern "C" {
    // stb_easy_font.h definiert diese Funktion:
    //   int stb_easy_font_print( float x, float y, char * text, unsigned char *color_rgb, void *vertex_buffer, int vbuf_size );
    int stb_easy_font_print(float x, float y, const char *text,
                            const unsigned char *color_rgb,
                            void *vertex_buffer, int vbuf_size);
}

namespace Hud {

    /// Zeichnet den kompletten HUD (FPS + Zoom/Offset).
    ///
    /// \param fps     aktuelle Frames‐pro‐Sekunde
    /// \param zoom    aktueller Zoom‐Faktor
    /// \param offsetX aktueller Offset.x
    /// \param offsetY aktueller Offset.y
    /// \param width   Fensterbreite (für Orthoprojektion)
    /// \param height  Fensterhöhe
    void draw(float fps,
              float zoom,
              float offsetX,
              float offsetY,
              int width,
              int height);

}

#endif // HUD_HPP
