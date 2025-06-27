// Datei: src/zoom_logic.cpp
// Zeilen: 83
// 🐭 Maus-Kommentar: Kapselt Zielauswahl rein auf relativer Basis. Bewertet Tiles nach Kontrast und Entropie-Gewinn. Schneefuchs: „Nur wer im Umfeld glitzert, zieht den Blick.“

#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <algorithm>

namespace ZoomLogic {

float computeEntropyContrast(const std::vector<float>& h, int index, int tilesX, int tilesY) {
    int x = index % tilesX;
    int y = index / tilesX;
    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < tilesX && ny >= 0 && ny < tilesY) {
                int nIndex = ny * tilesX + nx;
                sum += std::abs(h[index] - h[nIndex]);
                ++count;
            }
        }
    }
    return count > 0 ? sum / count : 0.0f;
}

ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    float2 offset,
    float zoom,
    int width,
    int height,
    int tileSize,
    RendererState& state
) {
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    float dynamicThreshold = std::max(Settings::VARIANCE_THRESHOLD / std::log2(zoom + 2.0f), Settings::MIN_VARIANCE_THRESHOLD);

    ZoomResult result{};
    result.bestIndex = -1;
    result.bestScore = -1.0f;

    for (int i = 0; i < numTiles; ++i) {
        int bx = i % tilesX;
        int by = i / tilesX;

        float centerX = (bx + 0.5f) * tileSize;
        float centerY = (by + 0.5f) * tileSize;

        float2 tileCenter = {
            (centerX - width / 2.0f) / zoom + offset.x,
            (centerY - height / 2.0f) / zoom + offset.y
        };

        float2 delta = { tileCenter.x - offset.x, tileCenter.y - offset.y };
        float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

        float contrast = computeEntropyContrast(entropy, i, tilesX, tilesY);
        float score = contrast > 0.0f ? entropy[i] / contrast : 0.0f;

        if (entropy[i] > dynamicThreshold && score > result.bestScore) {
            result.bestScore = score;
            result.newOffset = tileCenter;
            result.bestEntropy = entropy[i];
            result.bestContrast = contrast;
            result.bestIndex = i;
            result.distance = dist;
        }
    }

    constexpr float MIN_PIXEL_JUMP = 1.0f;
    result.minDistance = MIN_PIXEL_JUMP / zoom;

    result.relEntropyGain = (result.bestEntropy - state.lastEntropy) / (state.lastEntropy + 1e-6f);
    result.relContrastGain = (result.bestContrast - state.lastContrast) / (state.lastContrast + 1e-6f);

    result.isNewTarget = result.bestIndex >= 0 && (
        (result.relEntropyGain > 0.1f && result.relContrastGain > 0.05f) ||
        result.distance > result.minDistance
    );

    result.shouldZoom = result.bestIndex >= 0;

    return result;
}

} // namespace ZoomLogic
