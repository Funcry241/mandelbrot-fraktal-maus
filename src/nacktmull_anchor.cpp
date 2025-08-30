///// Otter: High-precision anchor orbit; fixed dz update to use z_n (correct derivative).
///// Schneefuchs: /WX-safe; added <cmath> for hypot/log; deterministic, ASCII-only comments.
///// Maus: Pure CPU; no hidden state; vectors pre-sized; Boost mp100.

#include "nacktmull_anchor.hpp"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <cmath>

namespace Nacktmull {

// Fixed 100-digit decimal float (compile-time precision).
using mp100 = boost::multiprecision::cpp_dec_float_100;

struct mp2 { mp100 x, y; };

static inline mp2 mul(const mp2& a, const mp2& b) {
    // (ax + i ay) * (bx + i by) = (ax*bx - ay*by) + i(ax*by + ay*bx)
    return { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
}

static inline mp2 add(const mp2& a, const mp2& b) { return { a.x + b.x, a.y + b.y }; }
static inline mp2 add_real(const mp2& a, const mp100& rx) { return { a.x + rx, a.y }; }

static inline double to_double(const mp100& v) { return static_cast<double>(v); }
static inline d2 to_d2(const mp2& v) { return { to_double(v.x), to_double(v.y) }; }

bool computeReferenceOrbit(const AnchorParams& params,
                           AnchorOrbit&       out,
                           int /*precDigitsHint*/, // kept for API symmetry
                           double /*bailoutRadius*/)
{
    const int N = std::max(0, params.maxIter);
    out.z.assign(static_cast<size_t>(N), d2{0.0, 0.0});
    out.dz.assign(static_cast<size_t>(N), d2{0.0, 0.0});
    out.maxIter = N;
    out.produced = 0;

    // c = centerX + i centerY (high precision).
    const mp2 c { mp100(params.centerX), mp100(params.centerY) };

    // z_0 = 0, dz_0/dc = 0
    mp2 z  { mp100(0), mp100(0) };
    mp2 dz { mp100(0), mp100(0) };

    for (int i = 0; i < N; ++i) {
        // Store current state (Z_i, dZ_i/dc) in double form.
        out.z[static_cast<size_t>(i)]  = to_d2(z);
        out.dz[static_cast<size_t>(i)] = to_d2(dz);
        ++out.produced;

        // Next iteration (use z_n for derivative update!):
        // z_{i+1}  = z_i^2 + c
        // dz_{i+1} = 2 * z_i * dz_i + 1
        const mp2 z_n = z;              // keep z_i
        const mp2 z2  = mul(z_n, z_n);  // z_i^2
        z  = add(z2, c);                // z_{i+1}

        const mp2 twoZ = { mp100(2) * z_n.x, mp100(2) * z_n.y };
        dz = add_real(mul(twoZ, dz), mp100(1));
    }

    return true;
}

bool shouldReuseAnchor(const AnchorParams& prev,
                       const AnchorParams& now,
                       int framebufferW,
                       int framebufferH,
                       double pixelTolerance)
{
    // Map complex delta to pixel units using the same framing as the renderer:
    // spanX = 3.5 / zoom; spanY = spanX * h / w; pixel size = span / dimension.
    if (framebufferW <= 0 || framebufferH <= 0) return false;

    const double spanX_prev = 3.5 / std::max(prev.zoom, 1e-30);
    const double spanY_prev = spanX_prev * (double)framebufferH / (double)framebufferW;

    const double pixX = spanX_prev / (double)framebufferW;
    const double pixY = spanY_prev / (double)framebufferH;

    const double dx = (now.centerX - prev.centerX) / std::max(pixX, 1e-300);
    const double dy = (now.centerY - prev.centerY) / std::max(pixY, 1e-300);
    const double distPixels = std::hypot(dx, dy);

    const double zoomRatio = now.zoom / std::max(prev.zoom, 1e-30);
    const double zoomDelta = std::abs(std::log(zoomRatio)); // ~relative change

    const bool centerOk = distPixels <= pixelTolerance;
    const bool zoomOk   = zoomDelta <= 0.01; // ≈ 1% change threshold

    return centerOk && zoomOk;
}

} // namespace Nacktmull
