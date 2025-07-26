// 🐭 Maus-Kommentar: Alpha 49f – Projekt „Frosch“: Technisch saubere Zoomlogik. Effizient, klar, schnell. Kein doppeltes Rechnen, keine unnötigen Branches. Otter: Glasklar. Schneefuchs: Makellos.

#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <iostream>
#include <chrono>

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    float2 currentOffset,
    float zoom,
    int width,
    int height,
    int tileSize,
    float2 previousOffset,
    [[maybe_unused]] int previousIndex,
    float previousEntropy,
    float previousContrast
) {
#ifndef __CUDA_ARCH__
    const auto t0 = std::chrono::high_resolution_clock::now(); // 🔬 Startzeit
#endif

    ZoomResult result;
    result.bestIndex = -1;
    result.shouldZoom = false;
    result.isNewTarget = false;
    result.newOffset = currentOffset;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int totalTiles = tilesX * tilesY;

    float bestScore = -1.0f;

    for (int i = 0; i < totalTiles; ++i) {
        const float e = entropy[i];
        const float c = contrast[i];
        if (e < Settings::ENTROPY_THRESHOLD_LOW) continue;

        const float score = e * (1.0f + c);
        if (score > bestScore) {
            bestScore = score;
            result.bestIndex = i;
            result.bestEntropy = e;
            result.bestContrast = c;
        }
    }

    if (result.bestIndex < 0)
        return result;

    const int bx = result.bestIndex % tilesX;
    const int by = result.bestIndex / tilesX;

    const float2 tileCenter = make_float2(
        (bx + 0.5f) * tileSize,
        (by + 0.5f) * tileSize
    );

    // in Clip-Space umrechnen [-1, 1]
    const float2 clipCenter = make_float2(
        (tileCenter.x / width - 0.5f) * 2.0f,
        (tileCenter.y / height - 0.5f) * 2.0f
    );

    const float2 proposedOffset = make_float2(
        currentOffset.x + clipCenter.x / zoom,
        currentOffset.y + clipCenter.y / zoom
    );

    const float dx = proposedOffset.x - previousOffset.x;
    const float dy = proposedOffset.y - previousOffset.y;
    const float dist = sqrtf(dx * dx + dy * dy);

    const float prevScore = previousEntropy * (1.0f + previousContrast);
    const float scoreGain = (prevScore > 0.0f) ? ((bestScore - prevScore) / prevScore) : 1.0f;

    result.isNewTarget = true;
    result.shouldZoom = true;

    const float alpha = Settings::ALPHA_LERP_MAX;
    result.newOffset = make_float2(
        previousOffset.x * (1.0f - alpha) + proposedOffset.x * alpha,
        previousOffset.y * (1.0f - alpha) + proposedOffset.y * alpha
    );

#ifndef __CUDA_ARCH__
    const auto t1 = std::chrono::high_resolution_clock::now(); // 🔬 Endzeit
    const float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (Settings::debugLogging) {
        std::printf("[ZoomEval] i=%d E=%.2f C=%.2f d=%.4f g=%.2f a=%.2f Z | %.3fms\n",
            result.bestIndex,
            result.bestEntropy,
            result.bestContrast,
            dist,
            scoreGain,
            alpha,
            ms
        );
    }
#endif

    return result;
}

} // namespace ZoomLogic
