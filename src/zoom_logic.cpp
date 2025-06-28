// Datei: src/zoom_logic.cpp
// Zeilen: 166
// 🐭 Maus-Kommentar: Umgestellt auf reine Datenübergabe – `RendererState` wird nicht mehr eingebunden. Damit ist diese Datei vollständig C++-kompatibel, auch für .cu-Dateien, falls gewünscht. Schneefuchs: „Trennung schafft Klarheit.“

#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <cstdio>

#define ENABLE_ZOOM_LOGGING 1

namespace ZoomLogic {

float computeEntropyContrast(float center, float neighbors[4]) {
    float contrast = 0.0f;
    for (int i = 0; i < 4; ++i) {
        contrast += fabsf(center - neighbors[i]);
    }
    return contrast / 4.0f;
}

float computeEntropyContrast(const std::vector<float>& h, int index, int tilesX, int tilesY) {
    int x = index % tilesX;
    int y = index / tilesX;
    float center = h[index];
    float neighbors[4] = {
        (x > 0         ? h[index - 1]     : center),
        (x < tilesX-1  ? h[index + 1]     : center),
        (y > 0         ? h[index - tilesX]: center),
        (y < tilesY-1  ? h[index + tilesX]: center)
    };
    return computeEntropyContrast(center, neighbors);
}

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
) {
    float dx = candidateTarget.x - currentTarget.x;
    float dy = candidateTarget.y - currentTarget.y;
    float dist = sqrtf(dx * dx + dy * dy);

    float minDist = Settings::MIN_JUMP_DISTANCE / zoom;
    float deltaEntropy = candidateEntropy - currentEntropy;
    float deltaContrast = candidateContrast - currentContrast;

    float relEntropy = currentEntropy > 0.01f ? deltaEntropy / currentEntropy : 0.0f;
    float relContrast = currentContrast > 0.01f ? deltaContrast / currentContrast : 0.0f;

    float dynamicScoreThreshold = 0.95f + 0.05f * log2f(zoom + 1.0f);
    bool skipByDelta = fabsf(deltaEntropy) < 0.3f && fabsf(deltaContrast) < 0.2f;

    isNewTarget = (dist > minDist && candidateScore > dynamicScoreThreshold && !skipByDelta);

    if (isNewTarget) {
        newTarget = candidateTarget;
    }

#if ENABLE_ZOOM_LOGGING
    printf(
        "ZoomLog Z %.5e Idx %d Ent %.5f S %.5f Dist %.6f Min %.6f dE %.4f dC %.4f RelE %.3f RelC %.3f dI %d New %d\n",
        zoom, candidateIndex, candidateEntropy, candidateScore,
        dist, minDist, deltaEntropy, deltaContrast,
        relEntropy, relContrast,
        (candidateIndex != currentIndex), isNewTarget
    );
#endif

    return isNewTarget;
}

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
) {
    ZoomResult result = {};
    result.bestScore = -1.0f;
    result.bestIndex = -1;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;

    for (int i = 0; i < tilesX * tilesY; ++i) {
        float entropy = h_entropy[i];
        float contrast = computeEntropyContrast(h_entropy, i, tilesX, tilesY);

        int x = i % tilesX;
        int y = i / tilesX;

        float2 candidateCenter;
        candidateCenter.x = static_cast<float>(offset.x + ((x + 0.5f) * tileSize - width / 2.0f) * zoom);
        candidateCenter.y = static_cast<float>(offset.y + ((y + 0.5f) * tileSize - height / 2.0f) * zoom);

        float score = entropy + contrast;

        float2 dummyNewTarget;
        bool isNew = false;

        bool accepted = selectZoomTarget(
            zoom, currentIndex, currentEntropy, currentContrast, currentOffset,
            candidateCenter, i, entropy, contrast, score,
            dummyNewTarget, isNew
        );

        if (accepted && score > result.bestScore) {
            result.bestScore = score;
            result.bestIndex = i;
            result.newOffset = make_double2(candidateCenter.x, candidateCenter.y);
            result.shouldZoom = true;
            result.bestEntropy = entropy;
            result.bestContrast = contrast;
            result.isNewTarget = isNew;

            float dx = candidateCenter.x - currentOffset.x;
            float dy = candidateCenter.y - currentOffset.y;
            result.distance = sqrtf(dx * dx + dy * dy);
            result.minDistance = Settings::MIN_JUMP_DISTANCE / zoom;
            result.relEntropyGain = currentEntropy > 0.01f ? (entropy - currentEntropy) / currentEntropy : 0.0f;
            result.relContrastGain = currentContrast > 0.01f ? (contrast - currentContrast) / currentContrast : 0.0f;
        }
    }

    return result;
}

} // namespace ZoomLogic
