///// Otter: Host Mandelbrot (CPU) declaration; returns ms and fills iteration buffer.
///// Schneefuchs: Signature matches cuda_interop; ABI stable; ASCII-only.
///// Maus: No logging; header/source in sync; [[nodiscard]] return.

#pragma once
#include <vector>

namespace Nacktmull {

// Computes CPU-side iteration counts for a full frame (width*height) into outIters.
// Returns elapsed milliseconds (double). ASCII-only, no logging.
[[nodiscard]] double compute_host_iterations(int width, int height,
                                             double zoom,
                                             double offX, double offY,
                                             int maxIter,
                                             std::vector<int>& outIters);

} // namespace Nacktmull
