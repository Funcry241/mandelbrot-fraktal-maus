///// Otter: CPU-Referenzrenderer für Nacktmull – klar, stabil, ohne Nebenpfade.
///// Schneefuchs: Deterministisch, ASCII-only; Schleifen leicht optimiert (dx/dy, Row-Ptr).
///// Maus: Keine Logs, kein noexcept auf Allokationspfaden; Header/Source synchron halten.

#include "nacktmull_types.hpp"
#include <vector>
#include <cstddef> // size_t

namespace nm {

struct View { nm::real cx, cy, spanX, spanY; int w, h; };

// Füllt "iters" (size = w*h) mit Escape-Iteration (oder maxIter wenn innen)
void render(const View& v, int maxIter, std::vector<int>& iters) {
    // Robustheit: negative/Null-Dimensionen oder maxIter<=0 → leere/Nullfläche
    const bool invalid = (v.w <= 0 || v.h <= 0 || maxIter <= 0);
    iters.assign(static_cast<size_t>(std::max(v.w, 0)) * static_cast<size_t>(std::max(v.h, 0)), 0);
    if (invalid) return;

    const nm::real two  = nm::real(2);
    const nm::real four = nm::real(4);

    // Precompute Mapping (Höhen-basierte Pixelgröße in X/Y)
    const nm::real x0 = v.cx - v.spanX * nm::real(0.5);
    const nm::real y0 = v.cy - v.spanY * nm::real(0.5);
    const nm::real dx = v.spanX / nm::real(v.w);
    const nm::real dy = v.spanY / nm::real(v.h);

    for (int y = 0; y < v.h; ++y) {
        const nm::real ci = (nm::real(y) + nm::real(0.5)) * dy + y0;
        int* __restrict row = iters.data() + static_cast<size_t>(y) * static_cast<size_t>(v.w);

        for (int x = 0; x < v.w; ++x) {
            const nm::real cr = (nm::real(x) + nm::real(0.5)) * dx + x0;

            nm::real zr = 0, zi = 0;
            int it = 0;

            for (; it < maxIter; ++it) {
                // z = z^2 + c
                const nm::real zr2 = zr * zr;
                const nm::real zi2 = zi * zi;

                // Escape-Test auf aktuellem z
                if (zr2 + zi2 > four) break;

                const nm::real zr_new = zr2 - zi2 + cr;
                zi = (two * zr * zi) + ci;
                zr = zr_new;
            }

            row[x] = it;
        }
    }
}

} // namespace nm
