// Datei: src/heatmap_overlay.hpp
// Zeilen: 30
/*
Maus-Kommentar 🐭: Nur relevante Schnittstellen bleiben – kein toter Code, kein Overhead. Overlay wird direkt per `drawOverlay(...)` gerendert. Schneefuchs: „Weniger ist manchmal Wärmebild.“
*/

#pragma once
#include <vector>

namespace HeatmapOverlay {

// Overlay an/aus (z. B. über Tastenevent)
void toggle();

// Gibt Ressourcen frei
void cleanup();

// Zeichnet das Overlay über das Fraktalbild
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width,
                 int height,
                 int tileSize,
                 GLuint textureId);

} // namespace HeatmapOverlay
