// Datei: src/core_kernel.h
// Zeilen: 32
#pragma once

#include <cuda_runtime.h>  // uchar4
#include <vector_types.h>   // float2

#ifdef __cplusplus
extern "C" {
#endif

// Fraktal-Rendering mit adaptivem Supersampling pro Tile
void launch_mandelbrotHybrid(
    uchar4* output,
    int* d_iterations,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    int tileSize,
    const int* d_tileSupersampling
);

// Entropie + Kontrast jedes Tiles berechnen (Host-Aufruf, implementiert in core_kernel.cu)
void computeCudaEntropyContrast(
    const int* d_iterations,
    float* d_entropyOut,
    float* d_contrastOut,
    int width,
    int height,
    int tileSize,
    int maxIter
);

#ifdef __cplusplus
}
#endif
