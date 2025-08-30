// ========================= src/nacktmull_anchor.hpp =========================
// Project Nacktmull — High-precision anchor orbit for Mandelbrot perturbation
// Host-only (no CUDA headers). Exposes a compact, GPU-friendly double layout.
// 🐭 Maus: ASCII-only, schlank; keine Nebenwirkungen im Header.
// 🦦 Otter: Rückgabewerte als [[nodiscard]] markiert (Fehler nicht übersehen).
// 🦊 Schneefuchs: Explizites 16-Byte-Alignment für CUDA-kompatibles Layout.

#pragma once

#include <vector>

namespace Nacktmull {

// 16-byte POD compatible with CUDA's double2 (layout: two doubles).
struct alignas(16) d2 {
    double x;
    double y;
};
static_assert(sizeof(d2) == 16, "d2 must be two doubles (16 bytes)");
static_assert(alignof(d2) == 16, "d2 must be 16-byte aligned");

// Parameters that define the current anchor (camera center).
struct AnchorParams {
    double centerX = 0.0;   // complex plane center (real)
    double centerY = 0.0;   // complex plane center (imag)
    double zoom    = 1.0;   // scene zoom (used for reuse heuristics)
    int    maxIter = 1000;  // reference orbit length to generate
};

// Output buffers for the reference orbit and its derivative.
struct AnchorOrbit {
    std::vector<d2> z;   // Z_n       for n = 0..maxIter-1
    std::vector<d2> dz;  // dZ/dc |_n for n = 0..maxIter-1
    int produced = 0;    // number of valid entries written (<= maxIter)
    int maxIter  = 0;    // requested length (echoed from params)
};

// Compute the high-precision reference orbit around the given center and
// write it into double buffers suitable for a CUDA kernel.
// Returns true on success; 'out' vectors are resized to params.maxIter.
//
// Notes:
//  * Precision uses Boost.Multiprecision cpp_dec_float_100 (≈ 100 decimal digits).
//  * We iterate up to params.maxIter unconditionally (no early bail).
//  * 'bailoutRadius' is for optional diagnostics; sequence remains complete.
[[nodiscard]] bool computeReferenceOrbit(const AnchorParams& params,
                                         AnchorOrbit&       out,
                                         int /*precDigitsHint*/ = 100,
                                         double bailoutRadius = 2.0);

// Heuristic to decide whether an existing anchor can be reused for a new
// camera state without recomputing the high-precision orbit.
//
// We map the delta in complex space to pixel units using current zoom and
// framebuffer size. If the center moved by less than 'pixelTolerance' and the
// zoom changed by <1% (fixed inside), we allow reuse.
[[nodiscard]] bool shouldReuseAnchor(const AnchorParams& prev,
                                     const AnchorParams& now,
                                     int framebufferW,
                                     int framebufferH,
                                     double pixelTolerance = 1.0);

} // namespace Nacktmull
