///// MAUS: header sync — entropy/contrast only (render kernels removed)
// Datei: src/core_kernel.h
// 🐭 Maus: Header minimal, eindeutig und stabil – keine heimlichen ABI-Änderungen.
// 🦊 Schneefuchs: Nur die tatsächlich implementierte API, C++-Linkage.

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cuda_runtime.h>   // cuda types / API
#include <vector_types.h>   // float2, uchar4 (nur Assertions)
#include <type_traits>

// Erwartete Plain-Old-Data-Eigenschaften (Hostsicht)
static_assert(std::is_trivial<float2>::value, "float2 must be trivial");
static_assert(std::is_trivial<uchar4>::value, "uchar4 must be trivial");
static_assert(sizeof(float2) == 8,            "float2 must be 8 bytes");
static_assert(sizeof(uchar4) == 4,            "uchar4 must be 4 bytes");

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
// Erwartet valide Größen/Allokationen gemäß width,height,tileSize.
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

#endif // CORE_KERNEL_H
