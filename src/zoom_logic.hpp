///// Otter: Schlanke Zoom-API; Hysterese-stabil; optionales Overlay-Material; ASCII-only.
///// Schneefuchs: Tiles (tilesX/tilesY) explizit vom Aufrufer; keine versteckte Geometrie; /WX-fest.
///// Maus: Eine klare API, ein Zustand, eine Entscheidung – deterministisch & instanzierbar.
///// Datei: src/zoom_logic.hpp

#pragma once

#include <vector>
#include <vector_types.h>   // float2

namespace ZoomLogic {

static_assert(sizeof(float2) == 8, "float2 must be 8 bytes");

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif

// Ergebnisstruktur fuer das Auto-Zoom-Ziel.
class ZoomResult {
public:
    int   bestIndex     = -1;    // Index des besten Tiles (Rasterindex)
    float bestEntropy   = 0.0f;  // Entropie dieses Tiles
    float bestContrast  = 0.0f;  // Kontrastwert
    float bestScore     = 0.0f;  // Gesamtscore (normalisiert, V2)

    float distance      = 0.0f;  // Entfernung zu previousOffset
    float minDistance   = 0.0f;  // Deadzone

    float relEntropyGain  = 0.0f;
    float relContrastGain = 0.0f;

    bool  isNewTarget   = false; // Zielwechsel akzeptiert?
    bool  shouldZoom    = false; // In diesem Frame zoomen?

    float2 newOffset    {0.0f, 0.0f}; // Zielkoordinate im Fraktalraum

    // Optionales Material fuer Overlay/Debug:
    std::vector<float> perTileContrast;
};

// Persistenter, minimaler Zustand des Zoomers (kein Global).
struct ZoomState {
    int   lastAcceptedIndex = -1;
    float lastAcceptedScore = 0.0f;
    int   cooldownLeft      = 0;
    float2 lastOffset       {0.0f, 0.0f};

    // Geometrie-Bookkeeping (Debug/Wechsel der Tile-Geometrie)
    int   lastTilesX        = -1;
    int   lastTilesY        = -1;
};

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

// (Optional) Kontrastanalyse ueber Tile-Nachbarn.
// Rueckgabe 0.0f bei unplausibler Geometrie.
[[nodiscard]] float computeEntropyContrast(
    const std::vector<float>& entropy,
    int width, int height, int tileSize) noexcept;

// Zoom V2 – eine API, eine Quelle der Wahrheit fuer Tiles.
//  - Entropie/Kontrast pro Frame normalisieren (median/MAD)
//  - Score = alpha*E' + beta*C'
//  - Hysterese & Cooldown stabilisieren die Zielwahl
//  - Offset-Glaettung (EMA), setzt shouldZoom
//  - Aktualisiert ZoomState in-place (kein Global)
[[nodiscard]] ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    int tilesX, int tilesY,
    int width, int height,
    float2 currentOffset, float zoom,
    float2 previousOffset,
    ZoomState& state) noexcept;

} // namespace ZoomLogic
