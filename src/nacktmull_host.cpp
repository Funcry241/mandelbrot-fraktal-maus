///// MAUS: host-side Mandelbrot iterations (implementation)
#include "nacktmull_host.hpp"
#include <chrono>
#include <cmath>

namespace Nacktmull {

static inline int mandelbrotIter(double cx, double cy, int maxIter, double bailoutSq)
{
    double zx = 0.0, zy = 0.0;
    int it = 0;
    while (it < maxIter) {
        const double zx2 = zx*zx - zy*zy + cx;
        const double zy2 = 2.0*zx*zy + cy;
        zx = zx2; zy = zy2;
        if (zx*zx + zy*zy > bailoutSq) break;
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

    const double invZoom = (zoom != 0.0) ? (1.0/zoom) : 1.0;
    const double ar      = (height != 0) ? static_cast<double>(width) / static_cast<double>(height) : 1.0;
    const double bailoutSq = 4.0;

    outIters.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    for (int y = 0; y < height; ++y) {
        const double ndcY = ((static_cast<double>(y) + 0.5) / static_cast<double>(height)) * 2.0 - 1.0;
        const double cy   = offY + ndcY * invZoom;
        const size_t row  = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = 0; x < width; ++x) {
            const double ndcX = ((static_cast<double>(x) + 0.5) / static_cast<double>(width)) * 2.0 - 1.0;
            const double cx   = offX + ndcX * invZoom * ar;
            outIters[row + static_cast<size_t>(x)] = mandelbrotIter(cx, cy, maxIter, bailoutSq);
        }
    }

    const auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace Nacktmull
