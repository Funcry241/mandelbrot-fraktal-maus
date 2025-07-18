// Datei: src/zoom_logic.cpp
// Zeilen: 81
// 🐅 Maus-Kommentar: Alpha 45b – Otterwunsch umgesetzt: Sobald ein Ziel mit ausreichend Entropie/Kontrast existiert, wird gezoomt, auch wenn der Index gleich bleibt. Kein Blockieren mehr durch isNewTarget. Schneefuchs: „Zoom ist eine Haltung."

#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>

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

    // 🧲 Otter: Immer zoomen, wenn Ziel interessant und nicht exakt gleich
    const float minMove = Settings::MIN_JUMP_DISTANCE / zoom;
    result.shouldZoom = (dist > minMove);

    return result;
}

} // namespace ZoomLogic
