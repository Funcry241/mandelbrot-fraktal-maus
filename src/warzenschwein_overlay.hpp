///// Otter: Projekt „Pfau“ – vereinheitlichte UI-Konstanten (Alpha/Margin/Radius) und Doku.
///// Schneefuchs: MAUS-Header (#62) strikt; ASCII-only; Header nur deklarativ, keine API-Änderung.
///// Maus: Pfau-Werte sind final verdrahtet (keine Alternativen); Pixel-Snapping-Helfer bereit.
///// Datei: src/warzenschwein_overlay.hpp

#pragma once
#include <string>
#include <vector>

namespace WarzenschweinOverlay {

// ─────────────────────────────────────────────────────────────────────────────
// Pfau-Thema: Einheitliche UI-Parameter (keine Alternativen, fest verdrahtet)
// ─────────────────────────────────────────────────────────────────────────────
namespace Pfau {
inline constexpr float UI_MARGIN   = 16.0f;   // Außenabstand zum Rand (px)
inline constexpr float UI_PADDING  = 12.0f;   // Innenabstand Text->Panelrand (px)
inline constexpr float UI_RADIUS   = 12.0f;   // Panel-Eckenradius (px)
inline constexpr float UI_BORDER   = 1.5f;    // Panel-Randstärke (px)
inline constexpr float PANEL_ALPHA = 0.84f;   // Panel-Deckkraft (0..1)
inline constexpr const char* LOG_PREFIX = "[UI/Pfau]"; // ASCII-Log-Präfix
// Anchor für Warzenschwein: **Top-Left** (Heatmap ist Top-Right)
enum class Anchor { TopLeft };
inline constexpr Anchor ANCHOR = Anchor::TopLeft;
} // namespace Pfau

// Pixel-Snapping-Helfer (ganzzahlige Device-Koordinaten; vermeidet Halbpixel-Weichzeichnung)
inline int snapToPixel(float v) { return static_cast<int>(v + 0.5f); }

// Overlay zeichnen – benötigt nur den aktuellen Zoom.
// Pfau-Regeln: Panel halbtransparent (PANEL_ALPHA), einheitliche Abstände/Radien,
// Blend: GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA (im .cpp gesetzt).
void drawOverlay(float zoom);

// Sichtbarkeit umschalten (internes Toggle).
void toggle();

// Textinhalt setzen (ASCII).
void setText(const std::string& text);

// OpenGL-Ressourcen freigeben (idempotent).
void cleanup();

// Erzeugt Vertex- und Hintergrund-Quads für die Textanzeige ohne RendererState.
// Viewport-Größe und Zoom werden explizit übergeben.
// Pfau-Anker: Top-Left, identische Top-Margin wie Heatmap (Top-Right) – symmetrische Höhe.
void generateOverlayQuads(
    const std::string& text,
    int viewportW,
    int viewportH,
    float zoom,
    std::vector<float>& vertexOut,
    std::vector<float>& backgroundOut);

} // namespace WarzenschweinOverlay
