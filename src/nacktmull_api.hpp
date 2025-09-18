///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: No forward-decls — include real types to avoid C4099.
///// Maus: Exact API; stream param explicit; stable include order.
///// Datei: src/nacktmull_api.hpp

#pragma once

#include <vector_types.h>        // float2, uchar4
#include <cstdint>
#include <cuda_runtime_api.h>    // cudaStream_t

// Konkrete Typen einbinden -> keine class/struct-Mismatches mehr:
#include "frame_context.hpp"
#include "renderer_state.hpp"

extern "C" {

// Progressive state setter (Keks 4/5)
void nacktmull_set_progressive(const void* zDev,
                               const void* itDev,
                               int addIter, int iterCap, int enabled);

// Unified Mandelbrot renderer (direct/progressive auto branch)
// Stream-Parameter explizit (C-ABI, keine Defaults)
void launch_mandelbrotHybrid(uchar4* out, uint16_t* d_it,
                             int w, int h, float zoom, float2 offset,
                             int maxIter, int tile,
                             cudaStream_t stream);

} // extern "C"

// ------------------------- C++ Host-API (Wrapper) -----------------------------
namespace NacktmullAPI {
    // Nutzt den vom RendererState besessenen renderStream (Schritt 4e).
    void computeCudaFrame(FrameContext& fctx, RendererState& state);
} // namespace NacktmullAPI
