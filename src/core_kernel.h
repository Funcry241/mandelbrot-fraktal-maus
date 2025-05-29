// Datei: src/core_kernel.h
// Maus-Kommentar: Deklariert Persistent-Mandelbrot-Kernel und Komplexitäts-Kernel für Auto-Zoom.

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cuda_runtime.h>
#include <vector_types.h>    // uchar4, float2

// Tile-Größe (muss mit core_kernel.cu übereinstimmen)
#define TILE_W 16
#define TILE_H 16

// Dynamic-Parallelism-Threshold für Nested-Kernel
#define DYNAMIC_THRESHOLD 100.0f

// Haupt-Kernel: Persistent Threads + Tile-Dispatch + Dynamic Parallelism
extern "C" __global__
void mandelbrotPersistent(
    uchar4* img,
    int     width,
    int     height,
    float   zoom,
    float2  offset,
    int     maxIter
);

// Komplexitäts-Kernel: Zählt nicht-schwarze Pixel pro Tile
extern "C" __global__
void computeComplexity(
    const uchar4* img,
    int           width,
    int           height,
    float*        complexity
);

#endif // CORE_KERNEL_H
