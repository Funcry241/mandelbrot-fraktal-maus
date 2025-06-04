#pragma once

#include <vector_types.h>   // für uchar4, float2

// ----------------------------------------------------------------------
// 1) Haupt-Kernel: Hybrid-Mandelbrot mit dynamischer Rekursion
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,        // Pointer auf Bild-Puffer (PBO-Mapping)
    int width,          // Bildbreite
    int height,         // Bildhöhe
    float zoom,         // aktueller Zoom-Faktor
    float2 offset,      // komplexer Offset (Pixel → Koordinate)
    int maxIter         // maximale Iterationsanzahl
);

// ----------------------------------------------------------------------
// 2) Debug-Gradient-Kernel (nur wenn DEBUG_GRADIENT definiert ist)
#ifdef DEBUG_GRADIENT
extern "C" void launch_debugGradient(
    uchar4* img,        // Pointer auf Bild-Puffer
    int width,          // Bildbreite
    int height          // Bildhöhe
);
#endif

// ----------------------------------------------------------------------
// 3) Complexity-Kernel: Zählt Pixel (Grauwert-Summe) pro Tile
__global__ void computeComplexity(
    const uchar4* img,  // bereits fertiggerendertes Bild (CUDA-Pointer auf PBO)
    int width,          // Bildbreite
    int height,         // Bildhöhe
    float* complexity   // Device-Array mit Länge (tilesX * tilesY)
);
