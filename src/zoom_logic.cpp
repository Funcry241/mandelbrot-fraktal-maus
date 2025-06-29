// Datei: src/zoom_logic.cpp
// Zeilen: 217
/*
Maus-Kommentar 🐭: Entscheidungskriterium für isNewTarget präzisiert. Statt schwammiger Schwellen nun klare, stufenweise Logik:
1. Ziel darf sich nicht nur in Index unterscheiden, sondern muss sich *qualitativ* lohnen (Score).
2. Entropie-Gewinn oder Kontrastgewinn allein reichen nicht mehr – der Score muss auch *signifikant* besser sein.
3. Das vermeidet Springen bei geringem Zoom-Gewinn. Schneefuchs-Vorgabe erfüllt: „Wenn du springst, dann mit Sinn.“
*/
#include "pch.hpp"
#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <cstdio>

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
    ZoomResult result;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int tileCount = tilesX * tilesY;

    result.bestIndex     = currentIndex;
    result.bestEntropy   = currentEntropy;
    result.bestContrast  = currentContrast;
    result.newOffset     = make_double2(currentOffset.x, currentOffset.y);
    result.perTileContrast.resize(tileCount, 0.0f);

    float maxScore = -1.0f;

    for (int i = 0; i < tileCount; ++i) {
        float entropy = h_entropy[i];

        int tx = i % tilesX;
        int ty = i / tilesX;

        float2 candidateOffset = {
            static_cast<float>(offset.x + tileSize * (tx + 0.5f - tilesX / 2.0f) / width),
            static_cast<float>(offset.y + tileSize * (ty + 0.5f - tilesY / 2.0f) / height)
        };

        float dx = candidateOffset.x - currentOffset.x;
        float dy = candidateOffset.y - currentOffset.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        float distWeight = 1.0f / (1.0f + dist * std::sqrt(zoom));
        float score = entropy * distWeight;

        std::printf("[ZOOMDBG] i %d tx %d ty %d score %.4f entropy %.4f dist %.6f offset %.6f %.6f\n",
                    i, tx, ty, score, entropy, dist, candidateOffset.x, candidateOffset.y);

        result.perTileContrast[i] = score;

        if (score > maxScore) {
            maxScore = score;
            result.bestIndex     = i;
            result.bestEntropy   = entropy;
            result.newOffset     = make_double2(candidateOffset.x, candidateOffset.y);
        }
    }

    float dx = static_cast<float>(result.newOffset.x - currentOffset.x);
    float dy = static_cast<float>(result.newOffset.y - currentOffset.y);
    float dist = std::sqrt(dx * dx + dy * dy);

    result.distance = dist;
    result.minDistance = Settings::MIN_JUMP_DISTANCE / zoom;
    result.relEntropyGain = result.bestEntropy - currentEntropy;
    result.relContrastGain = result.perTileContrast[result.bestIndex] - currentContrast;

    bool forcedSwitch = (result.perTileContrast[result.bestIndex] < 0.001f && result.distance > result.minDistance * 5.0f);

    // Neue Zielentscheidungslogik
    result.isNewTarget =
        (
            result.bestIndex != currentIndex &&
            result.perTileContrast[result.bestIndex] > currentContrast * 1.05f && // signifikanter Scoregewinn
            result.distance > result.minDistance
        )
        || forcedSwitch;

    result.shouldZoom = result.isNewTarget;

    return result;
}

} // namespace ZoomLogic
