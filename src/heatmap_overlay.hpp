///// Otter: Schlanker Header für Mini-Heatmap; klare API, keine GL-Abhängigkeiten im Header.
/// /// Schneefuchs: Nur Forward-Decls/Std-Types; kompatibel zu pch/GL-Includes im .cpp.
/// /// Maus: Beibehaltung der bestehenden Signaturen; deterministisch, übersichtlich.
/// /// Datei: src/heatmap_overlay.hpp
#pragma once

#include <vector>
#include "renderer_state.hpp"

// Kein GL-Header hier, damit Include-Reihenfolge (GLEW vor GL) nicht erzwungen wird.
using GLuint = unsigned int;

namespace HeatmapOverlay {

// Schaltet das Overlay-Flag im RendererState um (kein versteckter globaler Zustand).
void toggle(RendererState& ctx);

// Gibt alle vom Overlay angelegten GL-Ressourcen frei (idempotent).
void cleanup();

// Zeichnet die Mini-Heatmap unten rechts.
// Erwartet pro Tile genau einen Wert in 'entropy' und 'contrast' (Größe tilesX*tilesY).
// width/height = Framebuffergröße in Pixeln, tileSize = Kachelgröße in Pixeln.
// textureId wird aktuell nicht genutzt (Reserviert für künftige GPU-Pfade), darf 0 sein.
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 GLuint textureId,
                 RendererState& ctx);

} // namespace HeatmapOverlay
