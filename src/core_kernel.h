// 🐭 Maus-Kommentar: Header für CUDA-Kernel & Host-Launcher
// Definiert:
// - `launch_mandelbrotHybrid`: Fraktalberechnung + Iterationsspeicherung
// - `computeTileEntropy`: misst Entropie je Tile (zur Auto-Zoom-Steuerung)

#pragma once

#include <vector_types.h>  // für float2
#include <cuda_runtime.h>  // für __host__, __device__
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

// 🚀 Fraktal + Iterationen rendern (für spätere Analyse und Farbgebung)
void launch_mandelbrotHybrid(uchar4* output, int* d_iterations,
                             int width, int height,
                             float zoom, float2 offset,
                             int maxIterations);

// 📊 Entropie jedes Tiles berechnen (Iterationen → Verteilung → Entropie)
void computeTileEntropy(const int* d_iterations,
                        float* d_entropyOut,
                        int width, int height,
                        int tileSize,
                        int maxIter);

#ifdef __cplusplus
}
#endif
