// Datei: src/warzenschwein_overlay.hpp
// 🐭 Maus-Kommentar: Volle Kontrolle - Text wird extern via setText() gesetzt, Rest ist gekapselt. drawOverlay nutzt nur Zoom. Kein Zugriff auf internen RendererState nötig. Schneefuchs: Trennung, Otter: Lesbarkeit.

#pragma once
#include <string>
#include <vector>
#include "renderer_state.hpp"

namespace WarzenschweinOverlay {

// Overlay aktualisieren und zeichnen
void drawOverlay(RendererState&);

// Sichtbarkeit umschalten
void toggle(RendererState&);

// Textinhalt setzen
void setText(const std::string&);

// OpenGL-Resourcen freigeben
void cleanup();

// Erzeugt Vertex- und Hintergrunddaten für Textanzeige
void generateOverlayQuads(
    const std::string& text,
    std::vector<float>& vertexOut,
    std::vector<float>& backgroundOut,
    const RendererState& ctx);

} // namespace WarzenschweinOverlay

