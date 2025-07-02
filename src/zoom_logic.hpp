// 🐭 Maus-Kommentar: Nur noch Deklarationen! Für saubere Trennung von Interface und Implementation. CUDA-tauglich, kompakt. Schneefuchs: „Header macht Angebot, nicht Geschäft.“

#pragma once
#include "common.hpp"
#include "settings.hpp"
#include <vector>
#include <vector_types.h> // für double2

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

// 🧠 Berechnung: mittlerer Kontrast aus Nachbarentropien
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);

// 🧠 Entscheidung: neues Ziel auswählen (Panda-Version, 13 Argumente)
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    double2 currentOffset,
    float zoom,
    int width,
    int height,
    int tileSize,
    double2 previousOffset,
    int previousIndex,
    float previousEntropy,
    float previousContrast
);

} // namespace ZoomLogic
