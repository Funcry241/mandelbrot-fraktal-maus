#pragma once

#include <vector_types.h>   // Für uchar4, float2

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------
// 1) 🌀 Haupt-Kernel: Hybrid-Mandelbrot mit Iteration Buffer
void launch_mandelbrotHybrid(
    uchar4* img,         // 🖼️ Bildpuffer (CUDA/OpenGL-Interop)
    int* iterations,     // 🧠 Iterationspuffer (pro Pixel)
    int width,           // 📐 Bildbreite
    int height,          // 📐 Bildhöhe
    float zoom,          // 🔍 Zoom-Faktor
    float2 offset,       // 🎯 Offset im Fraktalraum
    int maxIter          // ⏳ Maximale Iterationszahl
);

// ----------------------------------------------------------------------
// 2) 🎨 Debug-Gradient-Kernel: Erzeugt Test-Farbverlauf
void launch_debugGradient(
    uchar4* img,         // 🖼️ Bildpuffer
    int width,           // 📐 Breite
    int height,          // 📐 Höhe
    float zoom           // 🔍 Zoom (für evtl. spätere Anpassungen)
);

// ----------------------------------------------------------------------
// 3) 🧠 Complexity-Kernel: Komplexitätsmessung (Standardabweichung in Tiles)
__global__ void computeComplexity(
    const int* iterations, // 🧠 Iterationspuffer
    int width,             // 📐 Breite
    int height,            // 📐 Höhe
    float* complexity,     // 📊 Ausgabe: Komplexität je Tile
    int tileSize           // 🧩 Dynamische Tile-Größe
);

// ----------------------------------------------------------------------
// 4) 🎯 Threshold Setter: Variance-Threshold setzen
void setDeviceVarianceThreshold(
    float threshold        // 🧮 Neuer Schwellenwert für Varianz
);

#ifdef __cplusplus
}
#endif
