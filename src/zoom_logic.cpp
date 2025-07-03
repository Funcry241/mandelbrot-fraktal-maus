// Datei: src/zoom_logic.cpp
// Zeilen: 140
// 🧠 Maus-Kommentar: Optimierte Panda-Version mit weniger Heap-Allokationen.
// Flugente-konform: Koordinaten sind wieder float-basiert (offset, zoom etc.).
// evaluateZoomTarget misst µs pro Frame – klar für Heatmap, Auto-Zoom und Debug-Auswertung.

#include "pch.hpp"
#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono> // ⏱ für Timing

#define ENABLE_ZOOM_LOGGING 0  // Set to 0 to disable local zoom analysis logs

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
    const std::vector<float>& h_contrast,
    float2 offset,              // Flugente: float2 statt double2
    float zoom,
    int width,
    int height,
    int tileSize,
    float2 currentOffset,       // Flugente: float2 statt double2
    int currentIndex,
    float currentEntropy,
    float currentContrast
) {
    auto t0 = std::chrono::high_resolution_clock::now(); // 🕒 Startzeit

    ZoomResult result;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int tileCount = tilesX * tilesY;

    result.bestIndex     = currentIndex;
    result.bestEntropy   = currentEntropy;
    result.bestContrast  = currentContrast;
    result.newOffset     = currentOffset;

    if (result.perTileContrast.capacity() < static_cast<size_t>(tileCount)) {
        result.perTileContrast.reserve(tileCount);
    }
    result.perTileContrast.assign(tileCount, 0.0f);

    float maxScore = -1.0f;

    for (int i = 0; i < tileCount; ++i) {
        float entropy = h_entropy[i];

        int tx = i % tilesX;
        int ty = i / tilesX;

        float2 candidateOffset = {
            offset.x + tileSize * (tx + 0.5f - tilesX / 2.0f) / zoom,
            offset.y + tileSize * (ty + 0.5f - tilesY / 2.0f) / zoom
        };

        float dx = candidateOffset.x - currentOffset.x;
        float dy = candidateOffset.y - currentOffset.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        float distWeight = 1.0f / (1.0f + dist * std::sqrt(zoom));
        float score = entropy * distWeight;

        result.perTileContrast[i] = score;

        if (score > maxScore) {
            maxScore = score;
            result.bestIndex     = i;
            result.bestEntropy   = entropy;
            result.bestContrast  = score;
            result.newOffset     = candidateOffset;
        }
    }

    float dx = result.newOffset.x - currentOffset.x;
    float dy = result.newOffset.y - currentOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    result.distance        = dist;
    result.minDistance     = Settings::MIN_JUMP_DISTANCE / zoom;
    result.relEntropyGain  = result.bestEntropy - currentEntropy;
    result.relContrastGain = result.perTileContrast[result.bestIndex] - currentContrast;
    result.bestScore       = result.perTileContrast[result.bestIndex];

    bool forcedSwitch = (result.bestScore < 0.001f && result.distance > result.minDistance * 5.0f);

    result.isNewTarget =
        (
            result.bestIndex != currentIndex &&
            result.bestScore > currentContrast * 1.05f &&
            result.distance > result.minDistance
        )
        || forcedSwitch;

    result.shouldZoom = result.isNewTarget || (result.distance >= Settings::DEADZONE);

#if ENABLE_ZOOM_LOGGING
    auto t1 = std::chrono::high_resolution_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::printf("[ZoomEval] idx=%d dE=%.4f dC=%.4f score=%.4f cur=%.4f dist=%.6f min=%.6f new=%d tiles=%d time=%lldus\n",
        result.bestIndex,
        result.relEntropyGain,
        result.relContrastGain,
        result.bestScore,
        currentContrast,
        result.distance,
        result.minDistance,
        result.isNewTarget ? 1 : 0,
        tileCount,
        micros
    );
#endif

    return result;
}

} // namespace ZoomLogic
