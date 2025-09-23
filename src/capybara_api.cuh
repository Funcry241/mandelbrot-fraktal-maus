///// Otter: Public API for Capybara Mandelbrot kernel (early Hi/Lo + classic continuation).
///// Schneefuchs: Header-only declaration; ASCII-only; no API break for existing code.
///// Maus: Include from host or device TU; provides launch prototype with cudaStream_t.
///// Datei: src/capybara_api.cuh

#pragma once
#include <stdint.h>

// Avoid heavy CUDA headers in public API: forward-declare cudaStream_t.
struct CUstream_st;
using cudaStream_t = CUstream_st*;

// Launches the Capybara Mandelbrot kernel that fills an iteration buffer (uint16_t per pixel).
// Parameters:
//   d_it    : device pointer to iteration output buffer of size w*h (uint16_t)
//   w, h    : image dimensions in pixels
//   cx, cy  : complex-plane center (double)
//   stepX   : complex step per pixel in X (double), typically scale / zoom
//   stepY   : complex step per pixel in Y (double) = stepX * (pixel_aspect_y / pixel_aspect_x)
//   maxIter : iteration cap (non-negative)
//   stream  : CUDA stream for asynchronous launch (may be nullptr)
//
// Notes:
// - Escape radius^2 is fixed at 4.0.
// - Uses Capybara early Hi/Lo iterations, then continues in double precision.
extern "C" void launch_mandelbrot_capybara(
    uint16_t* d_it,
    int w, int h,
    double cx, double cy,
    double stepX, double stepY,
    int maxIter,
    cudaStream_t stream /*= nullptr*/
);
