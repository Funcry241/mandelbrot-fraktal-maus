// Datei: src/core_kernel.h

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <vector_types.h> // uchar4, float2

// Tile-Dimension (muss zu main.cu passen)
#define TILE_W 16
#define TILE_H 16

// State‐of‐the‐Art Mandelbrot‐Kernel (separable Compilation)
extern "C" void launch_mandelbrotHybrid(uchar4* img,
                                        int w, int h,
                                        float zoom, float2 offset,
                                        int maxIter);

// Neuer Komplexitäts‐Kernel
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

#endif // CORE_KERNEL_H
