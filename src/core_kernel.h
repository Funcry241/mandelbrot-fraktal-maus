///// MAUS: header sync — Fast P1 + adaptive slice sizing documented (no API change)
// Datei: src/core_kernel.h
// 🐭 Maus: Header minimal, eindeutig und stabil – keine heimlichen ABI-Änderungen.
// 🦦 Otter: Klare Param-Doku, exakt die Signaturen wie in .cu verwendet. (Bezug zu Otter)
// 🦊 Schneefuchs: Name-Mangling-sicher via extern "C"-Block; Triviales Layout geprüft. (Bezug zu Schneefuchs)

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cuda_runtime.h>  // uchar4, cudaError_t etc.
#include <vector_types.h>  // float2
#include <type_traits>

// Erwartete Plain-Old-Data-Eigenschaften der CUDA Vektortypen (Hostsicht)
static_assert(std::is_trivial<float2>::value,  "float2 must be trivial");
static_assert(std::is_trivial<uchar4>::value,  "uchar4 must be trivial");
static_assert(sizeof(float2) == 8,             "float2 must be 8 bytes");
static_assert(sizeof(uchar4) == 4,             "uchar4 must be 4 bytes");

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// launch_mandelbrotHybrid
//
// Startet den hybriden Mandelbrot-Kernel (Bild + Iterationspuffer).
//  - devPtr       : Zielbild (OpenGL PBO, via CUDA-Interop gemappt), Format RGBA8
//  - d_iterations : Iterations-Count je Pixel (device memory, width*height ints)
//  - width/height : Bildabmessungen in Pixeln
//  - zoom         : Zoom-Faktor ( >0 ), 1.0 = Basisansicht
//  - offset       : Komplexe Verschiebung des Ausschnitts (Re = x, Im = y)
//  - maxIterations: Iterationslimit pro Pixel
//  - tileSize     : Kantenlänge der Tiles für die nachgelagerte Entropie/Contrast-Berechnung
//
// Hinweise:
//  - Keine Exceptions; Fehler werden im Hostpfad geloggt.
//  - Der Kernel schreibt ausschließlich in devPtr und d_iterations.
//  - 🦊 Schneefuchs: Analytischer Innen-Test (Cardioid + 2er-Bulb) direkt im Kernel.
//  - 🦦 Otter: Warmup & Slice-Strategie sind adaptiv, aber rein intern (keine API-/Header-Änderung).
//
void launch_mandelbrotHybrid(
    uchar4* devPtr,
    int*    d_iterations,
    int     width,
    int     height,
    float   zoom,
    float2  offset,
    int     maxIterations,
    int     tileSize
);

// ----------------------------------------------------------------------------
// computeCudaEntropyContrast
//
// Berechnet Entropie und lokalen Kontrast pro Tile aus dem Iterationspuffer.
//  - d_iterations : Iterations-Count je Pixel (device memory, width*height ints)
//  - d_entropyOut : Entropie je Tile (device memory, tilesX*tilesY floats)
//  - d_contrastOut: Kontrast je Tile (device memory, tilesX*tilesY floats)
//  - width/height : Bildabmessungen in Pixeln
//  - tileSize     : Kantenlänge der Tiles
//  - maxIterations: Iterationslimit (zur Histogramm-Normierung)
//
// Hinweise:
//  - Erwartet valide Sizes/Allokationen gemäß width,height,tileSize.
//  - Keine Host-Synchronisierung im Header – Implementierung verantwortet Sync.
//
void computeCudaEntropyContrast(
    const int* d_iterations,
    float*     d_entropyOut,
    float*     d_contrastOut,
    int        width,
    int        height,
    int        tileSize,
    int        maxIterations
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CORE_KERNEL_H
