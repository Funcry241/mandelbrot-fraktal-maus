// Datei: src/heatmap_overlay.hpp
// Zeilen: 31
/*
Maus-Kommentar 🐭: Nur relevante Schnittstellen bleiben – kein toter Code, kein Overhead. Overlay wird direkt per `drawOverlay(...)` gerendert. Schneefuchs: „Weniger ist manchmal Wärmebild.“
*/

#pragma once
#include <vector>
#include <GL/glew.h>

namespace HeatmapOverlay {

// Overlay ein-/ausblenden (z. B. via Tastendruck)
void toggle();

// Gibt GPU-Ressourcen (VAO, VBO, Shader) frei
void cleanup();

// Zeichnet das Debug-Overlay über dem Fraktalbild.
// entropy + contrast: Tile-Daten (gleiche Länge)
// width, height: Bildgröße in Pixel
// tileSize: Größe eines Tiles in Pixeln
// textureId: Fraktal-Textur (für optionales Blending)
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width,
                 int height,
                 int tileSize,
                 GLuint textureId);

} // namespace HeatmapOverlay
