///// Otter: Host-Mandelbrot (CPU) – schnelle Referenz mit inkrementeller NDC und Row-Ptr.
///// Schneefuchs: Deterministisch, ASCII-only; keine iostreams; Guards für Edge-Cases.
///// Maus: Keine versteckten Pfade; nur lokale Optimierungen, API unverändert.

#include "nacktmull_host.hpp"
#include <chrono>
#include <cmath>
#include <vector>
#include <cstddef>

namespace Nacktmull {

static inline int mandelbrotIter(double cx, double cy, int maxIter, double bailoutSq)
{
    double zx = 0.0, zy = 0.0;
    int it = 0;
    while (it < maxIter) {
        // z = z^2 + c
        const double zx_new = zx * zx - zy * zy + cx;
        const double zy_new = 2.0 * zx * zy + cy;
        zx = zx_new; zy = zy_new;

        const double r2 = zx * zx + zy * zy;
        if (r2 > bailoutSq) break;

        ++it;
    }
    return it;
}

double compute_host_iterations(int width, int height,
                               double zoom,
                               double offX, double offY,
                               int maxIter,
                               std::vector<int>& outIters)
{
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    // Edge-Cases: leere Fläche oder kein Budget → sofort zurück
    if (width <= 0 || height <= 0 || maxIter <= 0) {
        outIters.clear();
        const auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    outIters.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    // Vorberechnungen
    const double invZoom = (zoom != 0.0) ? (1.0 / zoom) : 1.0;
    const double ar      = static_cast<double>(width) / static_cast<double>(height);
    constexpr double bailoutSq = 4.0;

    // Inkrementelle NDC-Abbildung: ndc = a*x + b
    const double stepX = 2.0 / static_cast<double>(width);
    const double baseX = (1.0 / static_cast<double>(width)) - 1.0;   // x=0 → (0.5/width)*2 - 1
    const double stepY = 2.0 / static_cast<double>(height);
    const double baseY = (1.0 / static_cast<double>(height)) - 1.0;  // y=0 → (0.5/height)*2 - 1

    int* __restrict dst = outIters.data();

    for (int y = 0; y < height; ++y) {
        const double ndcY = baseY + stepY * static_cast<double>(y);
        const double cy   = offY + ndcY * invZoom;

        int* __restrict row = dst + static_cast<size_t>(y) * static_cast<size_t>(width);

        // x-Startwert pro Zeile und inkrementell laufen
        double ndcX = baseX;
        for (int x = 0; x < width; ++x, ndcX += stepX) {
            const double cx = offX + ndcX * invZoom * ar;
            row[x] = mandelbrotIter(cx, cy, maxIter, bailoutSq);
        }
    }

    const auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace Nacktmull
