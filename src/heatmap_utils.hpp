// Datei: src/heatmap_utils.hpp
// (Patch: noexcept + Guards – ASCII logs optional entfernen/aktivieren)
#pragma once
#include <utility> // std::pair
#include <cassert>

inline std::pair<double,double> tileIndexToPixelCenter(
    int tileIndex,
    int tilesX, int tilesY,
    int width, int height) noexcept
{
    assert(tilesX > 0 && tilesY > 0 && "tiles must be > 0");
    assert(width  > 0 && height > 0    && "image size must be > 0");

    // optional: clamp tileIndex in Release, damit wir nie UB riskieren
    const int total = tilesX * tilesY;
    if (tileIndex < 0)       tileIndex = 0;
    if (tileIndex >= total)  tileIndex = total - 1;

    const int tileX = tileIndex % tilesX;
    const int tileY = tileIndex / tilesX;

    const double tileW = static_cast<double>(width)  / static_cast<double>(tilesX);
    const double tileH = static_cast<double>(height) / static_cast<double>(tilesY);

    const double px = (static_cast<double>(tileX) + 0.5) * tileW;
    const double py = (static_cast<double>(tileY) + 0.5) * tileH;

    return {px, py};
}
