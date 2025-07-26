// Datei: src/warzenschwein_overlay.hpp
// 🐭 Maus-Kommentar: Volle Kontrolle – Text wird extern via setText() gesetzt, Rest ist gekapselt. drawOverlay nutzt nur Zoom. Kein Zugriff auf internen RendererState nötig. Schneefuchs: Trennung, Otter: Lesbarkeit.

#pragma once
#include "renderer_state.hpp"
#include <string>
#include <vector>

namespace WarzenschweinOverlay {

// ✏️ Setzt den darzustellenden Text (wird bei drawOverlay verwendet)
void setText(const std::string& text);

// ⬛ Wandelt den Text in Vertex- und Hintergrunddaten um (pro Frame neu generieren)
void generateOverlayQuads(
    const std::string& text,
    std::vector<float>& vertexOut,
    std::vector<float>& backgroundOut
);

// 🖼️ Zeichnet das Overlay (Text + Rechteck)
void drawOverlay(RendererState& ctx);

// 🔁 Aktiviert/Deaktiviert das Overlay (umschalten)
void toggle(RendererState& ctx);

// 🧹 Löscht OpenGL-Ressourcen (VAO/VBO etc.)
void cleanup();

} // namespace WarzenschweinOverlay
