// Datei: src/zoom_logic.hpp
// Zeilen: 66
// 🐭 Maus-Kommentar: Nur noch Deklarationen! Für saubere Trennung von Interface und Implementation. CUDA-tauglich, kompakt. Schneefuchs: „Header macht Angebot, nicht Geschäft.“

#pragma once
#include "common.hpp"
#include "settings.hpp"
#include <vector>

#define ENABLE_ZOOM_LOGGING 1

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

// Nur Deklarationen – Definitionen siehe .cpp
float computeEntropyContrast(float center, float neighbors[4]);
float computeEntropyContrast(const std::vector<float>& h, int index, int tilesX, int tilesY);

bool selectZoomTarget(
    float zoom,
    int currentIndex,
    float currentEntropy,
    float currentContrast,
    const float2& currentTarget,
    const float2& candidateTarget,
    int candidateIndex,
    float candidateEntropy,
    float candidateContrast,
    float candidateScore,
    float2& newTarget,
    bool& isNewTarget
);

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
