// Datei: src/core_kernel.h

// 🐭 Maus-Kommentar: Supersampling entfernt – launch_mandelbrotHybrid jetzt minimal und direkt.
// Otter: Klarer Fokus. Schneefuchs: Nur das, was gebraucht wird.

// 🌈 Rüsselwarze-Baustein: Farbcodierung basiert auf iterativen Escape-Pattern
// und strukturiertem Pseudozufall außerhalb des Fraktals.
// → Siehe core_kernel.cu: pseudoRandomWarze, hsvToRgb, mandelbrotKernel

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cuda_runtime.h>
#include <vector_types.h>

// Startet den hybriden Mandelbrot-Kernel (Bild + Iterationen)
// - devPtr: Zielbild (OpenGL PBO, gemappt auf CUDA)
// - d_iterations: Iterationsbuffer (pro Pixel)
// - width, height: Bildgröße
// - zoom, offset: Fraktalausschnitt
// - maxIterations: Iterationslimit
// - tileSize: Größe der Entropie-Tiles
// - Rüsselwarze: Innerhalb → dunkel. Außerhalb → strukturierter Zufall (radial, hue-basiert).
extern "C"
void launch_mandelbrotHybrid(
    uchar4* devPtr,
    int* d_iterations,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    int tileSize
);

// Entropie- und Kontrastberechnung pro Tile – Panda & Capybara
extern "C"
void computeCudaEntropyContrast(
    const int* d_iterations,
    float* d_entropyOut,
    float* d_contrastOut,
    int width,
    int height,
    int tileSize,
    int maxIterations
);

#endif // CORE_KERNEL_H
