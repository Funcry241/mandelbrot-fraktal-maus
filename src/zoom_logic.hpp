// MAUS:
// 🦦 Otter: Radikal vereinfacht – eine klare API, ein Zustand, eine Entscheidung. (Bezug zu Otter)
// 🦊 Schneefuchs: Explizite Tiles (tilesX/tilesY) werden vom Aufrufer geliefert; keine versteckte Geometrie. (Bezug zu Schneefuchs)
// 🐭 Maus: Keine Altlasten, keine Globals – deterministisch und instanzierbar.

#pragma once
#include "common.hpp"
#include <vector>
#include <vector_types.h> // float2

namespace ZoomLogic {

// 🛡️ Fallback für make_float2() – nur wenn CUDA-seitig nicht vorhanden
#ifndef CUDACC
[[nodiscard]] static inline float2 make_float2(float x, float y) {
    float2 f; f.x = x; f.y = y; return f;
}
#endif

static_assert(sizeof(float2) == 8, "float2 must be 8 bytes");

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324) // Padding durch float2-Ausrichtung – unkritisch
#endif

/// 🎯 Ergebnisstruktur für das Auto-Zoom-Ziel.
/// Wird pro Frame neu berechnet. Speichert Zielkoordinaten, Score und Analysewerte.
class ZoomResult {
public:
    int   bestIndex     = -1;    // Index des besten Tiles (Rasterindex)
    float bestEntropy   = 0.0f;  // Entropie dieses Tiles
    float bestContrast  = 0.0f;  // Kontrastwert
    float bestScore     = 0.0f;  // Gesamtscore (normalisiert, V2)

    float distance      = 0.0f;  // Entfernung des neuen Zielvorschlags zu previousOffset
    float minDistance   = 0.0f;  // Mindestabstand zur Zieländerung (Deadzone)

    float relEntropyGain  = 0.0f;
    float relContrastGain = 0.0f;

    bool  isNewTarget   = false; // Zielwechsel akzeptiert (Hysterese/Cooldown bestanden)?
    bool  shouldZoom    = false; // Soll in diesem Frame gezoomt werden?

    float2 newOffset    = make_float2(0.0f, 0.0f); // Zielkoordinaten im Fraktalraum
    std::vector<float> perTileContrast;            // optional (Overlay)
};

/// 🧭 Persistenter, minimaler Zustand des Zoomers (instanzierbar, kein Global).
struct ZoomState {
    int   lastAcceptedIndex = -1;
    float lastAcceptedScore = 0.0f;
    int   cooldownLeft      = 0;
    float2 lastOffset       = make_float2(0.0f, 0.0f);

    // Geometrie-Bookkeeping (hilft beim Debuggen / Wechsel der Tile-Geometrie)
    int   lastTilesX        = -1;
    int   lastTilesY        = -1;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/// 🐼 (Optional) Kontrastanalyse über Tile-Nachbarn.
/// Rückgabe 0.0 bei unplausibler Geometrie.
[[nodiscard]]
float computeEntropyContrast(
    const std::vector<float>& entropy,
    int width, int height, int tileSize) noexcept;

/// 🐘 Zoom V2 – eine API:
/// – Aufrufer liefert explizit tilesX/tilesY (eine Quelle für Tiles im System)
/// – normalisiert Entropie/Kontrast pro Frame (median/MAD)
/// – Score = αE' + βC'
/// – Hysterese & Cooldown stabilisieren Zielwahl
/// – glättet Offset-Bewegung (EMA) und setzt shouldZoom
/// – aktualisiert ZoomState in-place (kein Global)
[[nodiscard]]
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    int tilesX, int tilesY,
    int width, int height,
    float2 currentOffset, float zoom,
    float2 previousOffset,
    ZoomState& state) noexcept;

} // namespace ZoomLogic
