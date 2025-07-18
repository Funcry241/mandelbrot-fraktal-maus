// Datei: src/zoom_logic.cpp
// Zeilen: 120
// 🧠 Maus-Kommentar: Strikt C++23, keine Signed/Unsigned-Warnings mehr, keine unused-Variablen. Debug-Messungen sauber #if-gekappt. Otter- und Schneefuchs-konform.
#include "pch.hpp"
#include "zoom_logic.hpp"
#include "settings.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>

#define ENABLE_ZOOM_LOGGING 0

namespace ZoomLogic {

constexpr float SCORE_ALPHA = 0.5f, SCORE_BETA = 0.5f;

float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize) {
int tilesX = (width + tileSize - 1) / tileSize;
int tilesY = (height + tileSize - 1) / tileSize;
float maxC = 0.0f;
for (int y = 1; y < tilesY - 1; ++y)
for (int x = 1; x < tilesX - 1; ++x) {
int idx = y * tilesX + x;
float c = (std::abs(entropy[idx] - entropy[(y - 1) * tilesX + x]) +
std::abs(entropy[idx] - entropy[(y + 1) * tilesX + x]) +
std::abs(entropy[idx] - entropy[y * tilesX + (x - 1)]) +
std::abs(entropy[idx] - entropy[y * tilesX + (x + 1)])) / 4.0f;
maxC = std::max(maxC, c);
}
return maxC;
}

ZoomResult evaluateZoomTarget(
const std::vector<float>& h_entropy, const std::vector<float>& h_contrast,
float2 offset, float zoom, int width, int height, int tileSize,
float2 currentOffset, int currentIndex, float currentEntropy, float currentContrast
) {
#if ENABLE_ZOOM_LOGGING
auto t0 = std::chrono::high_resolution_clock::now();
#endif

int tilesX = (width + tileSize - 1) / tileSize;
int tilesY = (height + tileSize - 1) / tileSize;
int tileCount = tilesX * tilesY;

ZoomResult result;
result.bestIndex = currentIndex;
result.bestEntropy = currentEntropy;
result.bestContrast = currentContrast;
result.newOffset = currentOffset;

if (result.perTileContrast.capacity() < static_cast<size_t>(tileCount))
    result.perTileContrast.reserve(tileCount);
result.perTileContrast.assign(tileCount, 0.0f);

float maxScore = -1.0f;
for (int i = 0; i < tileCount; ++i) {
    float entropy = h_entropy[i];
    float contrast = (h_contrast.size() > static_cast<size_t>(i)) ? h_contrast[i] : 0.0f;
    int tx = i % tilesX, ty = i / tilesX;
    float2 cand = {
        offset.x + tileSize * (tx + 0.5f - tilesX / 2.0f) / zoom,
        offset.y + tileSize * (ty + 0.5f - tilesY / 2.0f) / zoom
    };
    float dx = cand.x - currentOffset.x, dy = cand.y - currentOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float distWeight = 1.0f / (1.0f + dist * std::sqrt(zoom));
    float score = (SCORE_ALPHA * entropy + SCORE_BETA * contrast) * distWeight;
    result.perTileContrast[i] = score;
    if (score > maxScore) {
        maxScore = score;
        result.bestIndex = i;
        result.bestEntropy = entropy;
        result.bestContrast = contrast;
        result.newOffset = cand;
    }
}

float dx = result.newOffset.x - currentOffset.x, dy = result.newOffset.y - currentOffset.y;
float dist = std::sqrt(dx * dx + dy * dy);
result.distance = dist;
result.minDistance = Settings::MIN_JUMP_DISTANCE / zoom;
result.relEntropyGain = result.bestEntropy - currentEntropy;
result.relContrastGain = result.bestContrast - currentContrast;
result.bestScore = result.perTileContrast[result.bestIndex];
bool forcedSwitch = (result.bestScore < 0.001f && dist > result.minDistance * 5.0f);

result.isNewTarget =
    ((result.bestIndex != currentIndex &&
      result.bestScore > currentContrast * 1.05f &&
      dist > result.minDistance)
     || forcedSwitch);

result.shouldZoom = result.isNewTarget || (dist >= Settings::DEADZONE);

#if ENABLE_ZOOM_LOGGING
auto t1 = std::chrono::high_resolution_clock::now();
auto micros = std::chrono::duration_caststd::chrono::microseconds(t1 - t0).count();
std::printf("[ZoomEval] idx=%d dE=%.4f dC=%.4f score=%.4f cur=%.4f dist=%.6f min=%.6f new=%d tiles=%d time=%lldus\n",
result.bestIndex, result.relEntropyGain, result.relContrastGain, result.bestScore, currentContrast,
result.distance, result.minDistance, result.isNewTarget ? 1 : 0, tileCount, micros);
#endif

return result;

}

} // namespace ZoomLogic
