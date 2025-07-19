// Datei: src/zoom_logic.cpp
// Zeilen: 112
// 🐅 Maus-Kommentar: Alpha 46c – Logging-Otter: Detailliertes Log zu Entropie, Kontrast, Offset-Distanz und Zoomschwelle. Diagnose bei Blockade. Schneefuchs: „Wer nicht zoom
#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <cfloat>
#include <iostream>

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
    int previousIndex,
    float previousEntropy [[maybe_unused]],
    float previousContrast [[maybe_unused]]
) {
    ZoomResult result;
    result.bestIndex = -1;
    result.bestEntropy = 0.0f;
    result.bestContrast = 0.0f;
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
        return result; // No target found

    int bx = result.bestIndex % tilesX;
    int by = result.bestIndex / tilesX;

    float2 tileCenter;
    tileCenter.x = (bx + 0.5f) * tileSize;
    tileCenter.y = (by + 0.5f) * tileSize;
    tileCenter.x = (tileCenter.x / width - 0.5f) * 2.0f;
    tileCenter.y = (tileCenter.y / height - 0.5f) * 2.0f;
    result.newOffset = make_float2(currentOffset.x + tileCenter.x / zoom,
                                   currentOffset.y + tileCenter.y / zoom);

    float dx = result.newOffset.x - previousOffset.x;
    float dy = result.newOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    result.isNewTarget = (result.bestIndex != previousIndex);

    const float minMove = Settings::MIN_JUMP_DISTANCE / zoom;
    bool offsetMoved = (dist > minMove);

    // 🔁 Immer weiter zoomen, solange das Ziel attraktiv bleibt – auch wenn Index gleich
    static float2 lastOffset = make_float2(FLT_MAX, FLT_MAX);
    static float lastZoom = -1.0f;

    constexpr float OFFSET_EPSILON = 1e-5f;
    constexpr float ZOOM_EPSILON   = 1e-4f;

    bool repeatedTarget = (std::abs(result.newOffset.x - lastOffset.x) < OFFSET_EPSILON) &&
                          (std::abs(result.newOffset.y - lastOffset.y) < OFFSET_EPSILON) &&
                          (std::abs(zoom - lastZoom) < ZOOM_EPSILON);

    if (!repeatedTarget || offsetMoved) {
        result.shouldZoom = true;
        lastOffset = result.newOffset;
        lastZoom = zoom;
    } else {
        result.shouldZoom = false;
    }

    // ASCII-only Log-Ausgabe zur Diagnose
    if (Settings::debugLogging) {
        std::cout << "[ZoomEval] idx=" << result.bestIndex
                  << " E=" << result.bestEntropy
                  << " C=" << result.bestContrast
                  << " dist=" << dist
                  << " jumpLimit=" << minMove
                  << " repeated=" << (repeatedTarget ? "1" : "0")
                  << " → zoom=" << (result.shouldZoom ? "1" : "0")
                  << "\n";
    }

    return result;
}

} // namespace ZoomLogic
