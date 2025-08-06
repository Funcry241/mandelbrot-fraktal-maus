// Datei: src/heatmap_utils.hpp
// Zweck: Gemeinsame Hilfsfunktionen für Heatmap-Overlay und Zoom-Logik
// 🦦 Otter: verhindert Drift, weil Overlay und Zoom dieselbe Umrechnung nutzen
// 🐭 Maus: zentrale Quelle für die ideale Spot-Berechnung
// 🐑 Schneefuchs: deterministisch, keine Seiteneffekte

#pragma once
#include <utility> // std::pair

// Berechnet den Pixelmittelpunkt einer Heatmap-Tile
// tilesX, tilesY: Anzahl Tiles in X/Y-Richtung
// width, height:  gesamte Bildabmessungen in Pixeln
// tileIndex:      Index der Tile (y * tilesX + x)
// Rückgabe:       (px, py) als double
inline std::pair<double,double> tileIndexToPixelCenter(
    int tileIndex,
    int tilesX, int tilesY,
    int width, int height)
{
    int tileX = tileIndex % tilesX;
    int tileY = tileIndex / tilesX;

    double actualTileSizeX = static_cast<double>(width)  / static_cast<double>(tilesX);
    double actualTileSizeY = static_cast<double>(height) / static_cast<double>(tilesY);

    double px = (static_cast<double>(tileX) + 0.5) * actualTileSizeX;
    double py = (static_cast<double>(tileY) + 0.5) * actualTileSizeY;

    return {px, py};
}
