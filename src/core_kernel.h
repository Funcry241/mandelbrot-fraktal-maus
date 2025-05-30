// Datei: src/core_kernel.h
#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <vector_types.h>  // float2
#include <vector_functions.h> // uchar4

// Tile-Größe
#define TILE_W 16
#define TILE_H 16

// --- Prototypen der CUDA-Kerne ---

// Haupt‐Kernel: Mandelbrot pro Tile, evtl. adaptiv in refineTile verzweigend
__global__ void mandelbrotHybrid(
    uchar4* img,
    int width, int height,
    float zoom, float2 offset,
    int maxIter);

// Nested‐Kernel zum Verfeinern
__global__ void refineTile(
    uchar4* img,
    int width, int height,
    float zoom, float2 offset,
    int startX, int startY,
    int tileW, int tileH,
    int maxIter);

// Hilfs‐Kernel zur Auswertung der Komplexität pro Tile
__global__ void computeComplexity(
    const uchar4* img,
    int width, int height,
    float* complexity);

// C‐API zum Aufruf des Haupt‐Kernels
extern "C"
void launch_mandelbrotHybrid(
    uchar4* img,
    int w, int h,
    float zoom, float2 offset,
    int maxIter);

#endif // CORE_KERNEL_H
