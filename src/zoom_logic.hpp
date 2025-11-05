///// Otter: Public API for Rullmolder Step 2 – blunt zoom + gentle nudge; dt-invariant; no capture deps.
///// Schneefuchs: Forward-Decls (FrameContext=struct, RendererState=struct), ASCII-only, /WX clean; C4099-frei.
///// Maus: API behält evaluateTarget() bei; zusätzlicher Wrapper evaluateZoomTarget() für Altaufrufe.
///// Datei: src/zoom_logic.hpp

#pragma once

#include <vector>
#include <cstddef>
#include <vector_types.h> // float2 in Funktionssignaturen

// Schlanke Forward-Decls (keine Includes hier, um Zyklen zu vermeiden)
struct FrameContext;   // definiert in frame_context.hpp  (struct)
struct RendererState;  // definiert in renderer_state.hpp (struct)

namespace ZoomLogic {

// Kleiner, trivially-constructible Zustand
struct ZoomState {
    bool hadCandidate = false;
};

// Ergebnis (ohne float2 in Structs → MSVC C4324-safe)
struct ZoomResult {
    float newOffsetX  = 0.0f;
    float newOffsetY  = 0.0f;
    float distance    = 0.0f;   // |newOffset - previousOffset|
    float minDistance = 0.02f;  // informativ
    int   bestIndex   = -1;     // Ziel-Tile oder -1
    bool  isNewTarget = false;
    bool  shouldZoom  = false;
};

// Kernzielwahl der Step-2-Pipeline (dein TU implementiert evaluateTarget)
ZoomResult evaluateTarget(const std::vector<float>& entropy,
                          const std::vector<float>& contrast,
                          int tilesX, int tilesY,
                          int width, int height,
                          float2 currentOffset, float zoom,
                          float2 previousOffset,
                          ZoomState& state) noexcept;

// Wrapper für Altcode, der noch evaluateZoomTarget(...) ruft
inline ZoomResult evaluateZoomTarget(const std::vector<float>& entropy,
                                     const std::vector<float>& contrast,
                                     int tilesX, int tilesY,
                                     int width, int height,
                                     float2 currentOffset, float zoom,
                                     float2 previousOffset,
                                     ZoomState& state) noexcept
{
    return evaluateTarget(entropy, contrast, tilesX, tilesY,
                          width, height, currentOffset, zoom, previousOffset, state);
}

// Pipeline-Adapter: schreibt pan/zoom direkt in RendererState.
// dtOverrideSeconds > 0.0 überschreibt fctx.deltaSeconds für genau diesen Aufruf.
void evaluateAndApply(::FrameContext& fctx,
                      ::RendererState& state,
                      ZoomState& bus,
                      float dtOverrideSeconds) noexcept;

} // namespace ZoomLogic
