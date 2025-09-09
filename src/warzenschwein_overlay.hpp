///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/warzenschwein_overlay.hpp
#pragma once
#include <string>
#include <vector>

namespace WarzenschweinOverlay {

// Overlay zeichnen – benoetigt nur den aktuellen Zoom.
// Alle weiteren Zustaende (Sichtbarkeit, Text, GPU-Ressourcen) liegen intern.
void drawOverlay(float zoom);

// Sichtbarkeit umschalten (internes Toggle).
void toggle();

// Textinhalt setzen (ASCII).
void setText(const std::string& text);

// OpenGL-Ressourcen freigeben (idempotent).
void cleanup();

// Erzeugt Vertex- und Hintergrund-Quads fuer die Textanzeige ohne RendererState.
// Viewport-Groesse und Zoom werden explizit uebergeben.
void generateOverlayQuads(
    const std::string& text,
    int viewportW,
    int viewportH,
    float zoom,
    std::vector<float>& vertexOut,
    std::vector<float>& backgroundOut);

} // namespace WarzenschweinOverlay
