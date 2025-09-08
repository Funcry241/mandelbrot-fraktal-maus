///// Otter: Shade helpers trimmed; only kernel prototypes kept; ASCII-only; CUDA13-ready.
///// Schneefuchs: Removed unused clamp01/make_rgba_01/hsv2rgb; launch_bounds kept; /WX-safe.
///// Maus: Minimal shading header; device-only intrinsics not exposed; stable signatures.
///*** Datei: src/nacktmull_shade.cuh

#pragma once

// CUDA / Vektor-Typen
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h> // make_float3, make_uchar4

// Standard-Math
#include <cmath>

// ------------------------ Kernel-Prototypen (Signaturen stabil) --------------
#if defined(__CUDACC__)
extern "C" __global__ __launch_bounds__(256,2)
void shade_from_iterations(uchar4* __restrict__ surface,
                           const int* __restrict__ iters,
                           int width, int height,
                           int maxIterations);

extern "C" __global__ __launch_bounds__(256,2)
void shade_test_pattern(uchar4* __restrict__ surface,
                        int width, int height,
                        float t);
#endif // __CUDACC__
