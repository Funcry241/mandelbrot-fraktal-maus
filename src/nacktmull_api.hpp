///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/nacktmull_api.hpp

#pragma once

#include <vector_types.h>  // float2, uchar4
#include <cstdint>
#include <cuda_runtime_api.h> // cudaStream_t

extern "C" {

// Progressive state setter (Keks 4/5)
void nacktmull_set_progressive(const void* zDev,
                               const void* itDev,
                               int addIter, int iterCap, int enabled);

// Unified Mandelbrot renderer (direct/progressive auto branch)
// NEU: expliziter Stream-Parameter (C-ABI, keine Defaults)
void launch_mandelbrotHybrid(uchar4* out, uint16_t* d_it,
                             int w, int h, float zoom, float2 offset,
                             int maxIter, int tile,
                             cudaStream_t stream);

} // extern "C"

// ------------------------- C++ Host-API (Wrapper) -----------------------------
class FrameContext;
class RendererState;

namespace NacktmullAPI {
    void computeCudaFrame(FrameContext& fctx, RendererState& state);
} // namespace NacktmullAPI
