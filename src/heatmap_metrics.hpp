///// Datei: src/heatmap_metrics.hpp
///// Otter: Thin header for GPU heatmap metrics.
///// Schneefuchs: Stable API; no heavy includes; ASCII-only.
///// Maus: Single entry point; stream-aware.

#pragma once

// Forward decls to keep header light
struct CUstream_st; using cudaStream_t = CUstream_st*;
class RendererState;

namespace HeatmapMetrics {

// Builds GPU metrics (boundary + contrast) into RendererState::h_entropy / h_contrast.
// Returns true on success. Synchronous on the given stream.
bool buildGPU(RendererState& state,
              int width, int height, int tilePx,
              cudaStream_t stream) noexcept;

} // namespace HeatmapMetrics
