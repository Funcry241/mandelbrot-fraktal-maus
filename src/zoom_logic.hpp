///// Otter: Public API for Silk-Lite auto-pan/zoom with robust drift fallback.
///// Schneefuchs: Header bleibt schlank; Forward-Decls statt schwerer Includes; /WX-fest.
///// Maus: Keine float2-Felder in Public-API-Strukturen (vermeidet MSVC C4324); ASCII-only.
// Datei: src/zoom_logic.hpp

#pragma once

#include <vector>
#include <cstddef>
#include <vector_types.h> // float2 in Funktionssignaturen

// Schlanke Forward-Decls (keine Zyklen):
struct FrameContext;     // definiert in frame_context.hpp (struct)
class  RendererState;    // definiert in renderer_state.hpp (class)

namespace ZoomLogic {

// Kleiner, trivially-constructible Zustand (by-value in RendererState erlaubt)
struct ZoomState {
    // Wird von evaluateAndApply gesetzt: true, wenn in diesem Frame ein valider
    // Zielkandidat (Tile) gefunden wurde. Die Pipeline nutzt das als Gate für den Zoom.
    bool hadCandidate = false;
};

// Ergebnis der Zielauswahl (ohne float2, um C4324 sicher zu vermeiden)
struct ZoomResult {
    float newOffsetX = 0.0f;
    float newOffsetY = 0.0f;
    float distance   = 0.0f;   // |newOffset - previousOffset|
    float minDistance= 0.02f;  // rein informativ
    int   bestIndex  = -1;     // Ziel-Tile oder -1
    bool  isNewTarget= false;
    bool  shouldZoom = false;
};

// Optionales Hilfsmaß (robuste Nachbarschaftsdifferenz)
float computeEntropyContrast(const std::vector<float>& entropy,
                             int width, int height, int tileSize) noexcept;

// Kern: bestes Ziel evaluieren und Bewegung vorschlagen
ZoomResult evaluateZoomTarget(const std::vector<float>& entropy,
                              const std::vector<float>& contrast,
                              int tilesX, int tilesY,
                              int width, int height,
                              float2 currentOffset, float zoom,
                              float2 previousOffset,
                              ZoomState& state) noexcept;

// Pipeline-Adapter: schreibt shouldZoom/newOffset in fctx
void evaluateAndApply(::FrameContext& fctx,
                      ::RendererState& state,
                      ZoomState& bus,
                      float gain) noexcept;

} // namespace ZoomLogic
