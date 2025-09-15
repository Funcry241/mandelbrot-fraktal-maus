///// Otter: Projekt „Pfau“ – identische UI-Parameter/Alpha; Anker Top-Right; Header bereinigt.
//\/\/ Schneefuchs: MAUS-Header (#62) strikt; keine GL-Includes im Header; ASCII-only Doku.
//\/\/ Maus: Pfau final; Funktionen unverändert, damit Build bis zur .cpp-Anpassung stabil bleibt.
///// Datei: src/heatmap_overlay.hpp
#pragma once

#include <vector>
#include "renderer_state.hpp"

// Kein GL-Header hier, damit Include-Reihenfolge (GLEW vor GL) nicht erzwungen wird.
using GLuint = unsigned int;

namespace HeatmapOverlay {

// ─────────────────────────────────────────────────────────────────────────────
// Pfau-Thema: Einheitliche UI-Parameter (keine Alternativen, fest verdrahtet)
// ─────────────────────────────────────────────────────────────────────────────
namespace Pfau {
inline constexpr float UI_MARGIN   = 16.0f;   // Außenabstand zum Rand (px)
inline constexpr float UI_PADDING  = 12.0f;   // Innenabstand Inhalt→Panelrand (px)
inline constexpr float UI_RADIUS   = 12.0f;   // Panel-Eckenradius (px)
inline constexpr float UI_BORDER   = 1.5f;    // Panel-Randstärke (px)
inline constexpr float PANEL_ALPHA = 0.84f;   // Panel-Deckkraft (0..1)
inline constexpr const char* LOG_PREFIX = "[UI/Pfau]"; // ASCII-Log-Präfix
// Anchor für Heatmap-Mini: **Top-Right** (WZ ist Top-Left)
enum class Anchor { TopRight };
inline constexpr Anchor ANCHOR = Anchor::TopRight;
} // namespace Pfau

// Pixel-Snapping-Helfer (für konsistente Kanten)
inline int snapToPixel(float v) { return static_cast<int>(v + 0.5f); }

// Schaltet das Overlay-Flag im RendererState um (kein versteckter globaler Zustand).
void toggle(RendererState& ctx);

// Gibt alle vom Overlay angelegten GL-Ressourcen frei (idempotent).
void cleanup();

// Zeichnet die Mini-Heatmap (Pfau-Material: halbtransparentes Panel mit identischem
// Margin/Padding/Radius wie Warzenschwein). Erwartet pro Tile genau einen Wert in
// 'entropy' und 'contrast' (Größe tilesX*tilesY). width/height = Framebuffergröße,
// tileSize = Kachelgröße in Pixeln. textureId aktuell optional (0 erlaubt).
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 GLuint textureId,
                 RendererState& ctx);

} // namespace HeatmapOverlay
