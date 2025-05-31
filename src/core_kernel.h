// Datei: src/core_kernel.h
#pragma once

#include <vector_types.h>   // für uchar4, float2

// ----------------------------------------------------------------------
// Tile‐Größe (muss in beiden Quelldateien identisch sein)
#ifndef TILE_W
#define TILE_W 16
#endif

#ifndef TILE_H
#define TILE_H 16
#endif

// ----------------------------------------------------------------------
// 1) Haupt‐Kernel: Hybrid‐Mandelbrot mit dynamischer Rekursion
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,        // Pointer auf Bild‐Puffer (PBO‐Mapping)
    int width,          // Bildbreite
    int height,         // Bildhöhe
    float zoom,         // aktueller Zoom‐Faktor
    float2 offset,      // komplexer Offset (Pixel → Koordinate)
    int maxIter         // maximale Iterationsanzahl
);

// ----------------------------------------------------------------------
// 2) Complexity‐Kernel: Zählt (z.B.) Pixel, die nicht schwarz sind
//    Pro Tile wird ein Float-Wert addiert (z.B. alle Pixel, deren Iterationscount < maxIter)
//    Damit kann der Host entscheiden, in welche Tile weiter gezoomt werden soll.
//
//    Achtung: Dieses Kernel‐Prototyp sollte __global__ sein, damit du in main.cu
//    bzw. core_kernel.cu darauf zugreifen kannst. Der Code wird in core_kernel.cu definiert.
extern "C" __global__ void computeComplexity(
    const uchar4* img,  // bereits fertiggerendertes Bild (CUDA‐Pointer auf PBO)
    int width,          // Bildbreite
    int height,         // Bildhöhe
    float* complexity   // Device‐Array mit Länge (tilesX * tilesY)
);

// ----------------------------------------------------------------------
// (Optional) Wenn es noch weitere Hilfs‐Funktionen oder Konstanten gibt, 
//   hier eintragen und dokumentieren.
// ----------------------------------------------------------------------
