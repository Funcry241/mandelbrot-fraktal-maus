// Datei: src/heatmap_overlay.hpp
// Zeilen: 47
/*
Maus-Kommentar 🐭: Diese Headerdatei verzichtet bewusst auf OpenGL-Includes und erwartet, dass `pch.hpp` bereits alles Notwendige bereitstellt (insbesondere `GLuint`). Das Overlay wird CUDA/OpenGL-basiert sein – ohne ImGui. Alle Funktionen sind explizit für die neue Textur-basierte Heatmap vorgesehen, Schneefuchs kann damit visuell validieren.
*/

#pragma once

#include <vector>

namespace HeatmapOverlay {

// Overlay an/aus (z. B. über Tastenevent)
void toggle();

// Initialisiert die Overlay-Textur bei Fenstergröße oder Tile-Änderung
void init(int width, int height);

// Gibt Ressourcen frei
void cleanup();

// Aktualisiert den Texturinhalt aus Entropie + Kontrast
void updateOverlayTexture(const std::vector<float>& entropy,
                          const std::vector<float>& contrast,
                          int width, int height,
                          int tileSize);

// Zeichnet das Overlay über das Fraktalbild
void drawOverlayTexture(int width, int height);

// Direkte Zeichnung mit Textur-ID (wird von renderer_loop.cpp verwendet)
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width,
                 int height,
                 int tileSize,
                 GLuint textureId);

} // namespace HeatmapOverlay
