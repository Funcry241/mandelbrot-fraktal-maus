// Datei: src/zoom_logic.hpp
// Zeilen: 41
// 🐭 Maus-Kommentar: Kompakt, nun korrekt sortiert. ZoomResult zuerst, dann Signaturen. Kein forward-declare nötig. Clang/CUDA-safe.
// 🦦 Otter: Effizienter Einstiegspunkt für die Zielauswahl – klar getrennte Verantwortlichkeiten.
// 🐅 Maus: Keine include-Kollisionen, keine Abhängigkeit zu math_utils, sauberes float2.

#pragma once
#include "common.hpp"
#include <vector>
#include <vector_types.h> // float2

namespace ZoomLogic {

// Struktur mit allen Informationen zum besten Zoomziel – wird bei jedem Frame ausgewertet.
struct ZoomResult {
    int   bestIndex = -1;
    float bestEntropy = 0.0f, bestContrast = 0.0f, bestScore = 0.0f;
    float distance = 0.0f, minDistance = 0.0f;
    float relEntropyGain = 0.0f, relContrastGain = 0.0f;
    bool  isNewTarget = false, shouldZoom = false;
    float2 newOffset = make_float2(0.0f, 0.0f);
    std::vector<float> perTileContrast; // Optionaler Rückkanal zur Heatmap oder Analyse
};

// 🐼 Panda: Entropie-Kontrastberechnung für jede Tile, mittelt 4er-Nachbarn
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);

// 🐘 + 🦦 + 🕊️ evaluateZoomTarget – Herzstück der Zoom-Entscheidung
// Liefert Zielkoordinaten und Bewertungsdaten für Auto-Zooming.
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    float2 currentOffset, float zoom,
    int width, int height, int tileSize,
    float2 previousOffset, int previousIndex,
    float previousEntropy, float previousContrast
);

} // namespace ZoomLogic
