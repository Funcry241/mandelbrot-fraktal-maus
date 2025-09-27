///// Otter: CUDA edge-detector API — samples visible grid, finds strongest edge (screen space).
///// Schneefuchs: Stable signature, ASCII-only logs elsewhere, minimal deps; noexcept-friendly.
///// Maus: One-kernel + host argmax; takes uint16 iteration buffer via RendererState.
// Datei: src/edge_detector.cuh

#pragma once
#include <cuda_runtime_api.h> // cudaStream_t

class RendererState;

namespace EdgeDetector {

struct Result {
    int   bestPx = -1;   // x in Pixeln (0..w-1)
    int   bestPy = -1;   // y in Pixeln (0..h-1)
    float grad  = 0.0f;  // Gradientenstärke (arbitrary units)
};

/// Sucht im Iterations-Buffer (state.d_iterations, uint16) die sichtbar stärkste
/// Kante anhand eines gleichmäßig verteilten Rasters.
/// - samplesX/samplesY: Rastergröße (empf. 8x8 oder 12x8)
/// - probeRadiusPx: Abstand der Finite-Difference-Samples (1..4)
/// Rückgabe: true bei Erfolg, out enthält Pixelposition & Grad.
bool findStrongestEdge(RendererState& state,
                       int width, int height,
                       int samplesX, int samplesY,
                       int probeRadiusPx,
                       cudaStream_t stream,
                       Result& out) noexcept;

} // namespace EdgeDetector
