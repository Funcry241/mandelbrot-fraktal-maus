// Datei: src/heatmap_overlay.hpp
/*
Maus-Kommentar 🐭: Overlay ist jetzt vollständig zustandslos - kein interner bool mehr. Alle Kontrollfunktionen arbeiten direkt mit RendererState&. drawOverlay-API akzeptiert ctx. Schneefuchs: „Kein Schatten, nur Klarheit.“ Otter: Kein struct/class-Konflikt mehr.
*/

#pragma once
#include <vector>

class RendererState;

namespace HeatmapOverlay {

// Overlay ein-/ausblenden via Tastendruck (setzt ctx.heatmapOverlayEnabled um)
void toggle(RendererState& ctx);

// Gibt GPU-Ressourcen (VAO, VBO, Shader) frei
void cleanup();

// 🦉 Projekt Eule: y=0 entspricht unterstem Bildrand.
// Die Heatmap-Daten (entropy/contrast) werden in Zeilen von unten nach oben interpretiert.
// drawOverlay() transformiert diese Tiles exakt wie ZoomLogic (kein Y-Flip).
// 🐑 Schneefuchs: „Kein vertikaler Schatten. Der Boden ist 0.“
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width,
                 int height,
                 int tileSize,
                 GLuint textureId,
                 RendererState& ctx);

} // namespace HeatmapOverlay
