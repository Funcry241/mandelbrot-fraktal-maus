// 🐅 Maus-Kommentar: Alpha 47a – Rückbau zu Variante A. Kein `lastOffset`-Hack mehr nötig: `ctx.offset` wird nun korrekt gesetzt. `shouldZoom` entscheidet nur auf Basis von Zielwechsel oder signifikanter Bewegung. Schneefuchs: „Wenn der Ort sich ändert, bewegt sich alles.“

#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
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

    // 🔍 Suche nach dem besten Tile anhand von Entropie und Kontrast
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

    // ❌ Kein geeignetes Ziel gefunden
    if (result.bestIndex < 0)
        return result;

    // 📍 Zielkoordinaten im Bild berechnen
    int bx = result.bestIndex % tilesX;
    int by = result.bestIndex / tilesX;

    float2 tileCenter;
    tileCenter.x = (bx + 0.5f) * tileSize;
    tileCenter.y = (by + 0.5f) * tileSize;
    tileCenter.x = (tileCenter.x / width - 0.5f) * 2.0f;
    tileCenter.y = (tileCenter.y / height - 0.5f) * 2.0f;

    result.newOffset = make_float2(
        currentOffset.x + tileCenter.x / zoom,
        currentOffset.y + tileCenter.y / zoom
    );

    // 📏 Bewegung berechnen
    float dx = result.newOffset.x - previousOffset.x;
    float dy = result.newOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    result.isNewTarget = (result.bestIndex != previousIndex);

    // 🧭 Zoom nur bei neuem Ziel oder spürbarer Bewegung
    const float minMove = Settings::MIN_JUMP_DISTANCE / zoom;
    result.shouldZoom = result.isNewTarget || (dist > minMove);

    // 🪵 ASCII-kompatibles Debug-Log
    if (Settings::debugLogging) {
        std::cout << "[ZoomEval] idx=" << result.bestIndex
                  << " E=" << result.bestEntropy
                  << " C=" << result.bestContrast
                  << " dist=" << dist
                  << " jumpLimit=" << minMove
                  << " new=" << (result.isNewTarget ? "1" : "0")
                  << " → zoom=" << (result.shouldZoom ? "1" : "0")
                  << "\n";
    }

    return result;
}

} // namespace ZoomLogic
