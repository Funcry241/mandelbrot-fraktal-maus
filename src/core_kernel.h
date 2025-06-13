#pragma once

#include <vector_types.h>   // Für CUDA-Typen wie uchar4, float2

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------
// 1) 🌀 Haupt-Kernel für das Mandelbrot-Rendering mit CUDA
//     → Hybrid-Ansatz: Farbwert & Iteration getrennt speicherbar
//     → Puffer: CUDA/OpenGL shared buffer (img), Iterationen (iterations)
void launch_mandelbrotHybrid(
    uchar4* img,         // 🖼️ Farbpuffer: 1 Pixel = 4 Byte (RGBA)
    int* iterations,     // 🔁 Iterationspuffer: Anzahl Schleifen je Pixel
    int width,           // 📐 Bildbreite in Pixel
    int height,          // 📐 Bildhöhe in Pixel
    float zoom,          // 🔍 Zoomfaktor (Pixel → Fraktalraum)
    float2 offset,       // 🎯 Mittelpunkt im Fraktalkoordinatensystem
    int maxIter          // ⏳ Maximal erlaubte Iterationen pro Punkt
);

// ----------------------------------------------------------------------
// 2) 🎨 Debug-Gradient-Kernel (Test-Rendering für GPU-Fehlersuche)
//     → Erzeugt horizontale + vertikale Farbverläufe
void launch_debugGradient(
    uchar4* img,         // 🧪 Farbpuffer für Debugausgabe
    int width,           // 📐 Bildbreite
    int height,          // 📐 Bildhöhe
    float zoom           // 🔍 Optionaler Zoom zur Skalierung
);

// ----------------------------------------------------------------------
// 3) 🧠 Komplexitätsanalyse (pro Kachel / Tile)
//     → Berechnet Standardabweichung je Tile (nicht pro Pixel!)
void computeComplexity(
    const int* iterations,  // 🧠 Iterationswerte aller Pixel
    float* stddev,          // σ Standardabweichung je Tile (device-Buffer)
    int width,              // 📐 Bildbreite in Pixel
    int height,             // 📐 Bildhöhe in Pixel
    int tileSize            // 🧩 Größe eines Tiles (z. B. 8x8, 16x16)
);

// ----------------------------------------------------------------------
// 4) 🎯 Dynamischer Schwellenwert für Komplexitätsbewertung
//     → Wird vom Host gesetzt, vom Device gelesen
void setDeviceVarianceThreshold(
    float threshold        // 🧮 Variance-Grenzwert zur Tile-Selektion
);

#ifdef __cplusplus
}
#endif
