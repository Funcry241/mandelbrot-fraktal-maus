///// Otter: Heatmap-Overlay-API – zustandslos, exakt wie ZoomLogic (kein Y-Flip).
///// Schneefuchs: Header/Source synchron, deterministisch, ASCII-only; keine verdeckten Pfade.
///// Maus: Nur LUCHS_LOG_* im Hostpfad; klare Parameter (Tiles, Texture, State).

#pragma once
#include <vector>
#include <GL/glew.h>  // für GLuint

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
