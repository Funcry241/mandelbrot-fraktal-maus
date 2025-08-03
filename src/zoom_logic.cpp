// Datei: src/zoom_logic.cpp
// Maus-Kommentar: Alpha 49 "Pinguin" - sanftes, kontinuierliches Zoomen ohne Elefant! Ziel wird immer interpoliert verfolgt, Score fließt in Glaettung ein. Kein Warten, kein Huepfen. Schneefuchs geniesst den Flug, Otter testet Stabilitaet.
// Panda: Bewertet Entropie x (1 + Kontrast) als Zielscore.
// Kolibri: Weiche Bewegung via LERP (Zoom ist Gleitflug).
// Flugente: float2 bleibt fuer Performance aktiv.
// Blaupause: Laufzeitmessung mit std::chrono - erkennt Zoomlogik-Overhead.

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <cmath>
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
    auto t0 = std::chrono::high_resolution_clock::now(); // Blaupause: Startzeit

    ZoomResult result;
    result.bestIndex = -1;
    result.shouldZoom = false;
    result.isNewTarget = false;
    result.newOffset = currentOffset;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const std::size_t totalTiles = static_cast<std::size_t>(tilesX * tilesY);

    if (Settings::debugLogging) {
        float minE = 9999.0f, maxE = -9999.0f;
        float minC = 9999.0f, maxC = -9999.0f;
        for (std::size_t i = 0; i < totalTiles; ++i) {
            if (i >= entropy.size() || i >= contrast.size()) continue;
            float e = entropy[i];
            float c = contrast[i];
            if (e < minE) minE = e;
            if (e > maxE) maxE = e;
            if (c < minC) minC = c;
            if (c > maxC) maxC = c;
        }
        LUCHS_LOG_HOST("[ZoomEval] Entropy: min=%.4f max=%.4f | Contrast: min=%.4f max=%.4f", minE, maxE, minC, maxC);
    }

    float bestScore = -1.0f;

    // Maus: Zugriffssicherheit hinzugefuegt - validiert Entropie-/Kontrastgroesse vor Zugriff
    for (std::size_t i = 0; i < totalTiles; ++i) {
        if (i >= entropy.size() || i >= contrast.size()) {
            LUCHS_LOG_HOST("[ZoomEval] Index %zu out of bounds (entropy=%zu, contrast=%zu)", i, entropy.size(), contrast.size());
            continue;
        }

        float entropyVal  = entropy[i];
        float contrastVal = contrast[i];

        float score = entropyVal * (1.0f + contrastVal);
        if (score > bestScore) {
            bestScore = score;
            result.bestIndex    = static_cast<int>(i);
            result.bestEntropy  = entropyVal;
            result.bestContrast = contrastVal;
        }
    }

    if (result.bestIndex < 0) {
        if (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZoomEval] No target found - bestScore=%.4f", bestScore);
        return result;
    }

    int bx = result.bestIndex % tilesX;
    int by = result.bestIndex / tilesX;

    float2 tileCenter;
    tileCenter.x = (bx + 0.5f) * tileSize;
    tileCenter.y = (by + 0.5f) * tileSize;
    tileCenter.x = (tileCenter.x / width  - 0.5f) * 2.0f;
    tileCenter.y = (tileCenter.y / height - 0.5f) * 2.0f;

    float2 proposedOffset = make_float2(
        currentOffset.x + tileCenter.x / zoom,
        currentOffset.y + tileCenter.y / zoom
    );

    float dx = proposedOffset.x - previousOffset.x;
    float dy = proposedOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    float prevScore = previousEntropy * (1.0f + previousContrast);
    float scoreGain = (prevScore > 0.0f) ? ((bestScore - prevScore) / prevScore) : 1.0f;

    result.isNewTarget  = true;
    result.shouldZoom   = true;

    float alpha = Settings::ALPHA_LERP_MAX; // Kolibri: Pinguin-Gleitflug - aggressiver LERP
    result.newOffset = make_float2(
        previousOffset.x * (1.0f - alpha) + proposedOffset.x * alpha,
        previousOffset.y * (1.0f - alpha) + proposedOffset.y * alpha
    );

    result.distance = dist;
    result.minDistance = Settings::MIN_JUMP_DISTANCE;
    result.relEntropyGain  = (result.bestEntropy > 0.0f && previousEntropy > 0.0f)
                             ? (result.bestEntropy - previousEntropy) / previousEntropy
                             : 1.0f;
    result.relContrastGain = (result.bestContrast > 0.0f && previousContrast > 0.0f)
                             ? (result.bestContrast - previousContrast) / previousContrast
                             : 1.0f;

    auto t1 = std::chrono::high_resolution_clock::now(); // Blaupause: Endzeit
    auto ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZoomEval] i=%d E=%.2f C=%.2f d=%.4f g=%.2f a=%.2f Z | %.3fms",
            result.bestIndex,
            result.bestEntropy,
            result.bestContrast,
            dist,
            scoreGain,
            alpha,
            ms
        );
    }

    return result;
}

} // namespace ZoomLogic
