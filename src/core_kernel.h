// Datei: src/core_kernel.h
// Maus-Kommentar: High-Performance Mandelbrot mit Persistent Threads, Tile-Blocking und Dynamic Parallelism.

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cuda_runtime.h>
#include <vector_types.h>     // für uchar4, float2
#include <vector_functions.h> // für make_uchar4, make_float2

// Globaler Zähler für Tiles (Persistent Kernel)
extern __device__ int tileIdxGlobal;

// Kachelgrößen
#define TILE_W 32
#define TILE_H 32

// Verfeinerungs-Kernel: führt bei "heißen" Kacheln doppelte Iterationszahl aus
__global__ void refineTile(uchar4* img,
                           int width, int height,
                           float zoom, float2 offset,
                           int startX, int startY,
                           int tileW, int tileH,
                           int maxIter);

// Haupt-Mandelbrot-Kernel mit Persistent Threads und Tile-Dispatch
__global__ void mandelbrotPersistent(uchar4* img,
                                     int width, int height,
                                     float zoom, float2 offset,
                                     int maxIter);

#endif // CORE_KERNEL_H
