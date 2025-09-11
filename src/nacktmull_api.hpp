///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/nacktmull_api.hpp
#pragma once

#include <vector_types.h>  // float2, uchar4

extern "C" {

// Progressive state setter (Keks 4/5)
// zDev: device pointer to float2[z] per pixel (resume state z)
// itDev: device pointer to int[it] per pixel   (resume iterations)
// addIter: iteration budget per frame
// iterCap: hard cap for iterations (usually maxIter of render call)
// enabled: 1 = progressive on, 0 = off (direct path)
void nacktmull_set_progressive(const void* zDev,
                               const void* itDev,
                               int addIter, int iterCap, int enabled);

// Unified Mandelbrot renderer (direct/progressive auto branch)
void launch_mandelbrotHybrid(uchar4* out, int* d_it,
                             int w, int h, float zoom, float2 offset,
                             int maxIter, int tile);

} // extern "C"
