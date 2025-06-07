#pragma once

#include <vector_types.h>   // für uchar4, float2

// ----------------------------------------------------------------------
// 1) Haupt-Kernel: Hybrid-Mandelbrot mit Iteration Buffer
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,        // Pointer auf Bild-Puffer (PBO-Mapping)
    int* iterations,    // 🐭 Buffer für Iterationszahlen
    int width,          // Bildbreite
    int height,         // Bildhöhe
    float zoom,         // aktueller Zoom-Faktor
    float2 offset,      // komplexer Offset (Pixel → Koordinate)
    int maxIter         // maximale Iterationsanzahl
);

// ----------------------------------------------------------------------
// 2) Debug-Gradient-Kernel: Erzeugt Test-Farbverlauf (nur bei debugGradient=true aktiv)
extern "C" void launch_debugGradient(
    uchar4* img,
    int width,
    int height
);

// ----------------------------------------------------------------------
// 3) Complexity-Kernel: Komplexitätsmessung auf Iterationspuffer
__global__ void computeComplexity(
    const int* iterations,  // 🐭 Iterationsbuffer statt fertiges Bild
    int width,              // Bildbreite
    int height,             // Bildhöhe
    float* complexity       // Device-Array mit Länge (tilesX * tilesY)
);
