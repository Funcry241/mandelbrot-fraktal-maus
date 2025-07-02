// Datei: src/zoom_logic.cpp
// Zeilen: 177
/*
👝 Maus-Kommentar: Fix laut Schneefuchs! Korrekte Umrechnung der Tile-Koordinaten nun zoom-basiert statt fensterbasiert. Kein „Springen“ mehr im Deep-Zoom. Undefinierte overloads entfernt. Logik jetzt stabil und glasklar. Jetzt auch: Kontrastwert im Zoom-Log sichtbar. Panda-Version: 13 Argumente.
*/

#include "pch.hpp"
#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace ZoomLogic {

float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize) {
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    float maxContrast = 0.0f;

    for (int y = 1; y < tilesY - 1; ++y) {
        for (int x = 1; x < tilesX - 1; ++x) {
            int idx = y * tilesX + x;
            float center = entropy[idx];
            float sumDiff = 0.0f;

            sumDiff += std::abs(center - entropy[(y - 1) * tilesX + x]);
            sumDiff += std::abs(center - entropy[(y + 1) * tilesX + x]);
            sumDiff += std::abs(center - entropy[y * tilesX + (x - 1)]);
            sumDiff += std::abs(center - entropy[y * tilesX + (x + 1)]);

            float contrast = sumDiff / 4.0f;
            maxContrast = std::max(maxContrast, contrast);
        }
    }

    return maxContrast;
}

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
) {
    ZoomResult result;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int tileCount = tilesX * tilesY;

    result.bestIndex     = previousIndex;
    result.bestEntropy   = previousEntropy;
    result.bestContrast  = previousContrast;
    result.newOffset     = previousOffset;
    result.perTileContrast.resize(tileCount, 0.0f);

    float maxScore = -1.0f;

    for (int i = 0; i < tileCount; ++i) {
        float e = entropy[i];
        float c = contrast[i];

        int tx = i % tilesX;
        int ty = i / tilesX;

        double2 candidateOffset = {
            currentOffset.x + tileSize * (tx + 0.5 - tilesX / 2.0) / zoom,
            currentOffset.y + tileSize * (ty + 0.5 - tilesY / 2.0) / zoom
        };

        double dx = candidateOffset.x - previousOffset.x;
        double dy = candidateOffset.y - previousOffset.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        float distWeight = 1.0f / (1.0f + static_cast<float>(dist * std::sqrt(zoom)));
        float score = e * distWeight;

#if ENABLE_ZOOM_LOGGING
        std::printf("[ZoomPick] i=%d tx=%d ty=%d score=%.4f entropy=%.4f dist=%.6f offset=(%.6f %.6f)%s\n",
            i, tx, ty, score, e, dist,
            candidateOffset.x, candidateOffset.y,
            (i == result.bestIndex ? " *BEST*" : "")
        );
#endif

        result.perTileContrast[i] = score;

        if (score > maxScore) {
            maxScore = score;
            result.bestIndex     = i;
            result.bestEntropy   = e;
            result.bestContrast  = c;
            result.newOffset     = candidateOffset;
        }
    }

    double dx = result.newOffset.x - previousOffset.x;
    double dy = result.newOffset.y - previousOffset.y;
    double dist = std::sqrt(dx * dx + dy * dy);

    result.distance = static_cast<float>(dist);
    result.minDistance = Settings::MIN_JUMP_DISTANCE / zoom;
    result.relEntropyGain = result.bestEntropy - previousEntropy;
    result.relContrastGain = result.bestContrast - previousContrast;
    result.bestScore = result.perTileContrast[result.bestIndex];

    bool forcedSwitch = (result.perTileContrast[result.bestIndex] < 0.001f && result.distance > result.minDistance * 5.0f);

    result.isNewTarget =
        (
            result.bestIndex != previousIndex &&
            result.perTileContrast[result.bestIndex] > previousContrast * 1.05f &&
            result.distance > result.minDistance
        )
        || forcedSwitch;

    if (!result.isNewTarget && result.distance >= Settings::DEADZONE) {
        result.shouldZoom = true;
    } else {
        result.shouldZoom = result.isNewTarget;
    }

#if ENABLE_ZOOM_LOGGING
    std::printf("[ZoomEval] idx=%d dE=%.4f dC=%.4f score=%.4f cur=%.4f dist=%.6f min=%.6f new=%d\n",
        result.bestIndex,
        result.relEntropyGain,
        result.relContrastGain,
        result.perTileContrast[result.bestIndex],
        previousContrast,
        result.distance,
        result.minDistance,
        result.isNewTarget ? 1 : 0
    );
#endif

    return result;
}

} // namespace ZoomLogic
