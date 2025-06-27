// Datei: src/zoom_logic.hpp
// Zeilen: 52
// 🐭 Maus-Kommentar: Deklariert die Zoom-Zielbewertungslogik getrennt von CUDA. Erlaubt saubere Trennung von Zustandsdaten und Zielauswahl. Schneefuchs: „Ein klarer Kopf entscheidet besser.“

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "renderer_state.hpp"

namespace ZoomLogic {

struct ZoomResult {
    float2 newOffset;
    bool shouldZoom;
    float bestScore;
    int bestIndex;
    float bestEntropy;
    float bestContrast;
    float distance;
    float minDistance;
    float relEntropyGain;
    float relContrastGain;
    bool isNewTarget;
};

ZoomResult evaluateZoomTarget(
    const std::vector<float>& h_entropy,
    float2 offset,
    float zoom,
    int width,
    int height,
    int tileSize,
    RendererState& state
);

float computeEntropyContrast(
    const std::vector<float>& h,
    int index,
    int tilesX,
    int tilesY
);

}  // namespace ZoomLogic
