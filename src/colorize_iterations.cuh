///// Otter: Iteration→PBO colorizer; one path, no dependencies on legacy files.
///// Schneefuchs: Deterministic HSV palette; interior = dark; ASCII-only, no device printf.
///// Maus: Simple API: colorize_iterations_to_pbo(d_it, pbo, w,h,maxIter,stream).
///// Datei: src/colorize_iterations.cuh

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Colorizes an iteration buffer (uint16 per pixel) into an RGBA8 image (uchar4)
// written directly to a mapped PBO. Interior pixels (it == maxIter) become dark.
// Launch is enqueued on `stream` and is non-blocking.
extern "C" void colorize_iterations_to_pbo(
    const uint16_t* d_iterations,
    uchar4*         d_pboOut,
    int             width,
    int             height,
    int             maxIter,
    cudaStream_t    stream
) noexcept;
