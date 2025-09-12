///// Otter: Header minimal und stabil - nur E/C-API; keine versteckten Abhaengigkeiten.
///// Schneefuchs: C++-Linkage, /WX-fest, ASCII-only; Header/Source synchron.
///// Maus: Render-Kernel entfernt; einzig verbleibende oeffentliche Funktion dokumentiert.
///// Datei: src/core_kernel.h

#pragma once
#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H
#include <cstdint>

// ----------------------------------------------------------------------------
// computeCudaEntropyContrast
//
// Berechnet Entropie und lokalen Kontrast pro Tile aus dem Iterationspuffer.
//  - d_iterations : Iterations-Count je Pixel (device memory, width*height ints)
//  - d_entropyOut : Entropie je Tile (device memory, tilesX*tilesY floats)
//  - d_contrastOut: Kontrast je Tile (device memory, tilesX*tilesY floats)
//  - width/height : Bildabmessungen in Pixeln
//  - tileSize     : Kantenlaenge der Tiles (Pixel)
//  - maxIterations: Iterationslimit (zur Histogramm-Normierung)
//
// Erwartet valide Groessen/Allokationen gemaess width,height,tileSize (C++-Linkage).
void computeCudaEntropyContrast(
    const uint16_t* d_iterations,
    float*     d_entropyOut,
    float*     d_contrastOut,
    int        width,
    int        height,
    int        tileSize,
    int        maxIterations
);

#endif // CORE_KERNEL_H
