// Datei: src/core_kernel.h
// Zeilen: 33
// 🐭 Maus-Kommentar: Schnittstelle zwischen CPU und CUDA-Kernel. Deklariert `launch_mandelbrotHybrid` (Fraktalrendering) und `computeTileEntropy` (Entropieanalyse pro Tile). Entfernt direkte CUDA-Includes, um Build-Probleme mit PCH und IntelliSense zu vermeiden. Schneefuchs sagte einst: „Die saubere Trennung macht den Unterschied.“

#pragma once

#include <vector_types.h>  // für float2

// 🧠 Kompatibilität für Host/Device-Makros in Nicht-CUDA-Kontexten
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 🚀 Fraktal + Iterationen rendern (für spätere Analyse und Farbgebung)
void launch_mandelbrotHybrid(uchar4* output, int* d_iterations,
                             int width, int height,
                             float zoom, float2 offset,
                             int maxIterations,
                             int supersampling);

// 📊 Entropie jedes Tiles berechnen (Iterationen → Verteilung → Entropie)
void computeTileEntropy(const int* d_iterations,
                        float* d_entropyOut,
                        int width, int height,
                        int tileSize,
                        int maxIter);

#ifdef __cplusplus
}
#endif
