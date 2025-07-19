// Datei: src/zoom_logic.cpp
// Zeilen: 162
// 🐅 Maus-Kommentar: Alpha 48.2 – `isTentativeNewTarget` nun aktiv genutzt für klare Debug-Markierung von Zielwechseln.
// 🐼 Panda: Bewertet Zielkacheln über Entropie × (1 + Kontrast).
// 🦦 Otter: Zoomdistanz abhängig vom Zoomlevel – weich geregelt.
// 🐘 Elefant: Ziel bleibt bestehen, wenn fast gleich gut – kein hektisches Springen.
// 🕊️ Kolibri: LERP gleicht sanft an – Zoom ist Bewegung, kein Ruck.
// 🐍 Flugente: float2 für Performance beibehalten.
// 🐑 Schneefuchs: TentativeFrames werden nur bei echten Zielwechseln zurückgesetzt.

#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <iostream>

namespace ZoomLogic {

template<typename T> inline T my_clamp(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

static int stableFrames = 0;
static int tentativeFrames = 0;
static int previousAcceptedIndex = -1;

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
        float e = entropy[i];
        float c = contrast[i];
        if (e < Settings::ENTROPY_THRESHOLD_LOW) continue;

        float score = e * (1.0f + c);
        if (score > bestScore) {
            bestScore = score;
            result.bestIndex = i;
            result.bestEntropy = e;
            result.bestContrast = c;
        }
    }

    if (result.bestIndex < 0)
        return result;

    int bx = result.bestIndex % tilesX;
    int by = result.bestIndex / tilesX;

    float2 tileCenter;
    tileCenter.x = (bx + 0.5f) * tileSize;
    tileCenter.y = (by + 0.5f) * tileSize;
    tileCenter.x = (tileCenter.x / width - 0.5f) * 2.0f;
    tileCenter.y = (tileCenter.y / height - 0.5f) * 2.0f;

    float2 proposedOffset = make_float2(
        currentOffset.x + tileCenter.x / zoom,
        currentOffset.y + tileCenter.y / zoom
    );

    float dx = proposedOffset.x - previousOffset.x;
    float dy = proposedOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float minMove = Settings::MIN_JUMP_DISTANCE / zoom;

    float prevScore = previousEntropy * (1.0f + previousContrast);
    float scoreGain = (prevScore > 0.0f) ? ((bestScore - prevScore) / prevScore) : 1.0f;
    float scoreDiff = (prevScore > 0.0f) ? std::abs(bestScore - prevScore) / prevScore : 1.0f;

    bool isTentativeNewTarget = (result.bestIndex != previousAcceptedIndex && scoreDiff > Settings::MIN_SCORE_DIFF_RATIO);
    result.isNewTarget = false;

    // 🐘 Stabilisierung mit Ziel-Gedächtnis
    if (result.bestIndex != previousAcceptedIndex) {
        if (scoreDiff > Settings::MIN_SCORE_DIFF_RATIO) {
            tentativeFrames = 1;
        } else {
            tentativeFrames = my_clamp(tentativeFrames + 1, 0, 1000);
        }
    } else {
        tentativeFrames = my_clamp(tentativeFrames + 1, 0, 1000);
    }

    bool isStableTarget = (tentativeFrames >= Settings::TENTATIVE_FRAMES_REQUIRED);
    result.isNewTarget = isStableTarget && (result.bestIndex != previousAcceptedIndex);

    if (isStableTarget) {
        previousAcceptedIndex = result.bestIndex;
    }

    int requiredStableFrames = my_clamp(static_cast<int>(Settings::MIN_STABLE_FRAMES + std::log2(zoom)),
                                        Settings::MIN_STABLE_FRAMES,
                                        Settings::MAX_STABLE_FRAMES);

    if (result.isNewTarget) {
        stableFrames = 0;
    } else {
        stableFrames = my_clamp(stableFrames + 1, 0, 1000);
    }

    result.shouldZoom = stableFrames >= requiredStableFrames && (dist > minMove || scoreGain > Settings::MIN_SCORE_GAIN_RATIO);

    float progress = my_clamp(static_cast<float>(stableFrames) / requiredStableFrames, 0.0f, 1.0f);
    float alpha = Settings::ALPHA_LERP_MIN + (Settings::ALPHA_LERP_MAX - Settings::ALPHA_LERP_MIN) * progress;

    result.newOffset = result.shouldZoom
        ? proposedOffset
        : make_float2(
            previousOffset.x * (1.0f - alpha) + proposedOffset.x * alpha,
            previousOffset.y * (1.0f - alpha) + proposedOffset.y * alpha);

    if (Settings::debugLogging) {
        std::cout << "[ZoomEval] idx=" << result.bestIndex
                  << " E=" << result.bestEntropy
                  << " C=" << result.bestContrast
                  << " dist=" << dist
                  << " jumpLimit=" << minMove
                  << " gain=" << scoreGain
                  << " scoreDiff=" << scoreDiff
                  << " tentative=" << tentativeFrames
                  << " stable=" << stableFrames
                  << " required=" << requiredStableFrames
                  << " alpha=" << alpha
                  << " isTentative=" << (isTentativeNewTarget ? "1" : "0")
                  << " new=" << (result.isNewTarget ? "1" : "0")
                  << " → zoom=" << (result.shouldZoom ? "1" : "0")
                  << "\n";
    }

    return result;
}

} // namespace ZoomLogic
