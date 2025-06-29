// Datei: src/zoom_logic.hpp
// Zeilen: 41
// 🐭 Maus-Kommentar: Nur noch Deklarationen! Für saubere Trennung von Interface und Implementation. CUDA-tauglich, kompakt. Schneefuchs: „Header macht Angebot, nicht Geschäft.“

#pragma once
#include "common.hpp"
#include "settings.hpp"
#include <vector>

namespace ZoomLogic {

struct ZoomResult {
    int bestIndex = -1;
    float bestEntropy = 0.0f;
    float bestContrast = 0.0f;
    float bestScore = 0.0f;
    float distance = 0.0f;
    float minDistance = 0.0f;
    float relEntropyGain = 0.0f;
    float relContrastGain = 0.0f;
    bool isNewTarget = false;
    bool shouldZoom = false;
    double2 newOffset = make_double2(0.0, 0.0);

    std::vector<float> perTileContrast;  // 🔥 Kontrastwerte für HeatmapOverlay
};

// Kontrastberechnung aus Nachbarentropien (nur eine definierte Version bleibt)
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);

// Hauptentscheidung: neues Zoom-Ziel ja/nein
ZoomResult evaluateZoomTarget(
    const std::vector<float>& h_entropy,
    double2 offset,
    double zoom,
    int width,
    int height,
    int tileSize,
    float2 currentOffset,
    int currentIndex,
    float currentEntropy,
    float currentContrast
);

} // namespace ZoomLogic
