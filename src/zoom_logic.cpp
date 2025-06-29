// Datei: src/zoom_logic.cpp
// Zeilen: 162
/*
Maus-Kommentar 🐭: Diese Datei wurde irrtümlich als Ort für Hauptfunktionen wie `renderFrame`, `drawFrame` etc. verwendet – das führt zu symbolischen Duplikaten mit `renderer_loop.cpp`. Schneefuchs sagt: „Nie zweimal das Gleiche rufen lassen, sonst knallt der Linker.“
Diese Datei ist jetzt korrekt bereinigt und enthält **ausschließlich** logische Auswertungsfunktionen wie Entropiekontrast, Zoom-Bewertung und Heatmap-Werte.
*/

#include "pch.hpp"
#include "zoom_logic.hpp"
#include "settings.hpp"

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

        float distWeight = 1.0f / (1.0f + dist * zoom);
        float score = entropy * distWeight;

        result.perTileContrast[i] = score;  // Heatmap-Wert

        if (score > maxScore) {
            maxScore = score;
            result.bestIndex     = i;
            result.bestEntropy   = entropy;
            result.newOffset     = make_double2(candidateOffset.x, candidateOffset.y);
        }
    }

    // Entscheidung über Zoom-Ziel
    float dx = static_cast<float>(result.newOffset.x - currentOffset.x);
    float dy = static_cast<float>(result.newOffset.y - currentOffset.y);
    float dist = std::sqrt(dx * dx + dy * dy);

    result.distance = dist;
    result.minDistance = Settings::MIN_JUMP_DISTANCE / zoom;
    result.relEntropyGain = result.bestEntropy - currentEntropy;
    result.relContrastGain = result.perTileContrast[result.bestIndex] - currentContrast;

    result.isNewTarget =
        result.bestIndex != currentIndex &&
        result.relEntropyGain > 0.01f &&
        result.relContrastGain > 0.01f &&
        result.distance > result.minDistance;

        
    result.isNewTarget = true;
    

    result.shouldZoom = result.isNewTarget;

    return result;
}

} // namespace ZoomLogic
