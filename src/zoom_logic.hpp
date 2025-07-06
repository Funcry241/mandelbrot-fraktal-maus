// Datei: src/zoom_logic.hpp
// Zeilen: 39
// 🐭 Maus-Kommentar: Kompakt, rein deklarativ. Nur das Interface für Hotspot-Auswertung und Scoring. Flugente/Capybara-ready, Header bleibt minimal – Schneefuchs-Nivea
#pragma once
#include "common.hpp"
#include <vector>
#include <vector_types.h> // float2

namespace ZoomLogic {

struct ZoomResult {
    int   bestIndex = -1;
    float bestEntropy = 0.0f, bestContrast = 0.0f, bestScore = 0.0f;
    float distance = 0.0f, minDistance = 0.0f;
    float relEntropyGain = 0.0f, relContrastGain = 0.0f;
    bool  isNewTarget = false, shouldZoom = false;
    float2 newOffset = make_float2(0.0f, 0.0f);
    std::vector<float> perTileContrast;
};

// Nachbarkontrast: Mittelwert aus 4er-Nachbarschaft
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);

// Auswahl des Zoom-Hotspots (Entropie+Kontrast, 13 Args)
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    float2 offset, float zoom,
    int width, int height, int tileSize,
    float2 currentOffset, int currentIndex,
    float currentEntropy, float currentContrast
);

} // namespace ZoomLogic
