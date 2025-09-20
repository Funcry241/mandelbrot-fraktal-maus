///// Otter: Deterministic host-side reference-orbit builder; double-precision only.
///// Schneefuchs: No hidden side effects; header-only API honored; zero device interaction.
///// Maus: ASCII-only; no logs by default; safe early-stop on non-finite values.
///// Datei: src/perturbation_orbit.cpp

#include "perturbation_orbit.hpp"

#include <cmath>     // std::isfinite

// Build z_{n+1} = z_n^2 + c in double precision starting at z_0 = 0.
// Writes z_1..z_len into `out`. Deterministic for fixed (c, maxLen, segSize).
void buildReferenceOrbit(const double2 c,
                         int maxLen,
                         int /*segSize*/,
                         std::vector<double2>& out,
                         int& len)
{
    // Guard contracts (robustness without throwing for negative inputs).
    const int N = (maxLen > 0) ? maxLen : 0;

    out.clear();
    out.reserve(static_cast<std::size_t>(N));

    // z_0 = 0
    double zx = 0.0;
    double zy = 0.0;

    // Generate up to N samples; stop early if non-finite appears.
    for (int i = 0; i < N; ++i)
    {
        const double x = zx;
        const double y = zy;

        // z^2 + c with minimal temporaries for determinism
        const double nx = x * x - y * y + c.x;
        const double ny = (x + x) * y + c.y;

        // Early-out if arithmetic diverges to non-finite values
        if (!std::isfinite(nx) || !std::isfinite(ny))
        {
            break;
        }

        out.push_back(double2{nx, ny});
        zx = nx;
        zy = ny;
    }

    len = static_cast<int>(out.size());
    // Vector size already equals len due to push_back loop; capacity may exceed, which is fine.
}
