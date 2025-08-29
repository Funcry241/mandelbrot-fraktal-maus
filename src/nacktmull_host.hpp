///// MAUS: host-side Mandelbrot iterations (declaration)
#pragma once
// 🦊 Schneefuchs: Genau die Signatur, die in cuda_interop.cu aufgerufen wird. (Bezug zu Schneefuchs)
#include <vector>

namespace Nacktmull {

// Computes CPU-side iteration counts for a full frame (width*height) into outIters.
// Returns elapsed milliseconds (double). ASCII-only, no logging.
double compute_host_iterations(int width, int height,
                               double zoom,
                               double offX, double offY,
                               int maxIter,
                               std::vector<int>& outIters);

} // namespace Nacktmull
