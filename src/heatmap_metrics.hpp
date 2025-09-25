///// Otter: GPU heatmap metrics (entropy/contrast) — one public entry point.
///// Schneefuchs: Minimal includes; ASCII-only logs; no GL; stable signature.
///// Maus: Slab-allocated device buffers; single-kernel path; noexcept API.
///// Datei: src/heatmap_metrics.hpp

#pragma once
#include <cuda_runtime_api.h>  // cudaStream_t

class RendererState;

namespace HeatmapMetrics {

/// Compute entropy/contrast per tile on the GPU and mirror to state.h_*.
/// Returns false on error (logged via Luchs).
bool buildGPU(RendererState& state,
              int width, int height, int tilePx,
              cudaStream_t stream) noexcept;

} // namespace HeatmapMetrics
