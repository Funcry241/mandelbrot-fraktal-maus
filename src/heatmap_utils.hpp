// Datei: src/heatmap_utils.hpp
// Zweck: Gemeinsame Hilfsfunktionen für Heatmap-Overlay und Zoom-Logik
// 🦦 Otter: verhindert Drift, weil Overlay und Zoom dieselbe Umrechnung nutzen
// 🐭 Maus: zentrale Quelle für die ideale Spot-Berechnung
// 🐑 Schneefuchs: deterministisch, keine Seiteneffekte

#pragma once
#include <utility> // std::pair

// Liefert den Pixelmittelpunkt einer Heatmap-Tile.
// 🦉 Projekt Eule: y=0 ist die **unterste** Kachelreihe.
// Kachelindexierung erfolgt zeilenweise: tileIndex = y * tilesX + x
// Ergebnis ist in Bildschirmkoordinaten (Pixelmitte), nicht NDC oder Complex.
// Diese Funktion wird konsistent von Zoom-Logik und Heatmap-Overlay verwendet.
// 🐑 Schneefuchs: Die Kacheln wachsen von unten nach oben, deterministisch.
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
