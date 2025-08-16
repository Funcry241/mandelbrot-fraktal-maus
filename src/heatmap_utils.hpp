// MAUS:
// Datei: src/heatmap_utils.hpp
// 🐭 Maus: Korrekte Kachel-Geometrie ohne Rundungsfehler; half-open Bounds [x0,x1)×[y0,y1).
// 🦦 Otter: Keine Exceptions, klare Guards, noexcept überall. (Bezug zu Otter)
// 🐑 Schneefuchs: Center via Integer-Bounds (teilbare und nicht-teilbare Auflösungen stabil). (Bezug zu Schneefuchs)

#pragma once
#include <utility>   // std::pair
#include <algorithm> // std::clamp
#include <cassert>

// Interne Helfer: integer-sichere Boundsteilung (half-open).
// Garantiert vollständige Abdeckung des Bildes auch bei width%tilesX != 0 etc.
inline void tileCoordsToPixelBounds(
    int tileX, int tileY,
    int tilesX, int tilesY,
    int width,  int height,
    int& x0, int& y0, int& x1, int& y1) noexcept
{
    assert(tilesX > 0 && tilesY > 0 && "tiles must be > 0");
    assert(width  > 0 && height > 0    && "image size must be > 0");

    // Clamp Koordinaten (Release-Schutz; Debug fängt assert).
    tileX = std::clamp(tileX, 0, tilesX - 1);
    tileY = std::clamp(tileY, 0, tilesY - 1);

    // Integer-Splitting vermeidet kumulative FP-Rundungsfehler.
    x0 = ( tileX      * width)  / tilesX;
    x1 = ((tileX + 1) * width)  / tilesX;
    y0 = ( tileY      * height) / tilesY;
    y1 = ((tileY + 1) * height) / tilesY;
}

// Wandelt TileIndex → (tileX,tileY).
inline std::pair<int,int> tileIndexToCoords(
    int tileIndex, int tilesX, int tilesY) noexcept
{
    const int total = tilesX * tilesY;
    if (tilesX <= 0 || tilesY <= 0 || total <= 0) return {0,0};
    // Clamp Index zur Sicherheit.
    if (tileIndex < 0)        tileIndex = 0;
    if (tileIndex >= total)   tileIndex = total - 1;
    return { tileIndex % tilesX, tileIndex / tilesX };
}

// Half-open Pixel-Bounds eines Tiles als ints.
inline void tileIndexToPixelBounds(
    int tileIndex,
    int tilesX, int tilesY,
    int width, int height,
    int& x0, int& y0, int& x1, int& y1) noexcept
{
    auto [tx, ty] = tileIndexToCoords(tileIndex, tilesX, tilesY);
    tileCoordsToPixelBounds(tx, ty, tilesX, tilesY, width, height, x0, y0, x1, y1);
}

// Mittelpunkt eines Tiles in Pixelkoordinaten (double, zur Subpixel-Darstellung).
// Hinweis: Center wird aus integeren Bounds abgeleitet → robust bei Restkacheln.
inline std::pair<double,double> tileIndexToPixelCenter(
    int tileIndex,
    int tilesX, int tilesY,
    int width, int height) noexcept
{
    assert(tilesX > 0 && tilesY > 0 && "tiles must be > 0");
    assert(width  > 0 && height > 0    && "image size must be > 0");

    if (tilesX <= 0 || tilesY <= 0 || width <= 0 || height <= 0) {
        return {0.0, 0.0};
    }

    int x0, y0, x1, y1;
    tileIndexToPixelBounds(tileIndex, tilesX, tilesY, width, height, x0, y0, x1, y1);

    const double cx = 0.5 * (static_cast<double>(x0) + static_cast<double>(x1));
    const double cy = 0.5 * (static_cast<double>(y0) + static_cast<double>(y1));
    return {cx, cy};
}

// Integer-Mittelpunkt (für Pixel-gerechte Marker ohne Subpixel-Blend).
inline std::pair<int,int> tileIndexToPixelCenterInt(
    int tileIndex,
    int tilesX, int tilesY,
    int width, int height) noexcept
{
    int x0, y0, x1, y1;
    tileIndexToPixelBounds(tileIndex, tilesX, tilesY, width, height, x0, y0, x1, y1);
    // floor((x0+x1)/2) via Integer (overflow-safe bei int-Range).
    return { (x0 + x1) >> 1, (y0 + y1) >> 1 };
}

// Pixel → Tile-Coords (tileX,tileY), basierend auf half-open Bounds.
// px/py außerhalb des Bildes werden an den gültigen Bereich geklemmt.
inline std::pair<int,int> pixelToTileCoords(
    int px, int py,
    int tilesX, int tilesY,
    int width, int height) noexcept
{
    if (tilesX <= 0 || tilesY <= 0 || width <= 0 || height <= 0) return {0,0};
    // Clamp Pixel in sichtbaren Bereich [0..W-1],[0..H-1]
    px = std::clamp(px, 0, std::max(0, width  - 1));
    py = std::clamp(py, 0, std::max(0, height - 1));
    const int tx = std::min(tilesX - 1, (px * tilesX) / width);
    const int ty = std::min(tilesY - 1, (py * tilesY) / height);
    return {tx, ty};
}

// Pixel → TileIndex (bequemer Wrapper).
inline int pixelToTileIndex(
    int px, int py,
    int tilesX, int tilesY,
    int width, int height) noexcept
{
    auto [tx, ty] = pixelToTileCoords(px, py, tilesX, tilesY, width, height);
    return ty * tilesX + tx;
}
