// Datei: src/core_kernel.h
// Zeilen: 34
// 🐭 Maus-Kommentar: Schnittstelle zwischen CPU und CUDA-Kernel. Jetzt mit Panda: `computeEntropyContrast` liefert Entropie und Kontrast pro Tile – für Heatmap und Zielwahl. Schneefuchs sagte: „Struktur ist der Anfang von Neugier.“

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

// 🐼 Entropie + Kontrast jedes Tiles berechnen (für Heatmap + Zielwahl)
void computeEntropyContrast(const int* d_iterations,
                            float* d_entropyOut,
                            float* d_contrastOut,
                            int width, int height,
                            int tileSize,
                            int maxIter);

#ifdef __cplusplus
}
#endif
