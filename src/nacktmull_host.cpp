#include "nacktmull_host.hpp"
#include <chrono>
#include <cmath>
#include <vector>
#include <cstddef>

namespace Nacktmull {

// Likely-inline Kernel: minimal Arithmetik, FMA wo sinnvoll (Host-CPU)
static inline int mandelbrotIter(double cx, double cy, int maxIter, double bailoutSq)
{
    double zx = 0.0, zy = 0.0;
    int it = 0;

    while (it < maxIter) {
        // z' = z^2 + c
        // Berechne neues z aus altem z (x=zx, y=zy)
        const double x2 = zx * zx;
        const double y2 = zy * zy;

        // zx' = (x^2 - y^2) + cx
        const double zx_new = (x2 - y2) + cx;

        // zy' = 2*x*y + cy  → per FMA: (zx+zx)*zy + cy  (eine Mul weniger, identisches Ergebnis)
        const double zy_new = std::fma(zx + zx, zy, cy);

        // |z'|^2  → FMA reduziert Rundungsfehler und Instruktionen
        const double r2 = std::fma(zx_new, zx_new, zy_new * zy_new);
        if (r2 > bailoutSq) break;

        zx = zx_new;
        zy = zy_new;
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

    // Edge-Cases
    if (width <= 0 || height <= 0 || maxIter <= 0) {
        outIters.clear();
        const auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    outIters.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    // Vorberechnungen (alles double, deterministisch)
    const double invZoom = (zoom != 0.0) ? (1.0 / zoom) : 1.0;
    const double ar      = static_cast<double>(width) / static_cast<double>(height);
    constexpr double bailoutSq = 4.0;

    // Inkrementelle Abbildung ohne per-Pixel-Divisionen:
    // ndcX = (2/w)*(x+0.5) - 1  → cx = offX + ndcX * invZoom * ar
    // Wir laufen direkt über cx inkrementell:
    const double stepX_ndc = 2.0 / static_cast<double>(width);
    const double baseX_ndc = (1.0 / static_cast<double>(width)) - 1.0; // x=0 → (0.5/w)*2 - 1
    const double cx_step   = stepX_ndc * invZoom * ar;
    const double cx_row0   = offX + baseX_ndc * invZoom * ar;

    // Für Y bleibt es pro Zeile konstant:
    const double stepY_ndc = 2.0 / static_cast<double>(height);
    const double baseY_ndc = (1.0 / static_cast<double>(height)) - 1.0;

    int* __restrict dst = outIters.data();

    for (int y = 0; y < height; ++y) {
        // cy für diese Zeile
        const double ndcY = baseY_ndc + stepY_ndc * static_cast<double>(y);
        const double cy   = offY + ndcY * invZoom;

        // Zeilenzeiger + inkrementeller cx
        int* __restrict row = dst + static_cast<size_t>(y) * static_cast<size_t>(width);
        double cx = cx_row0;

        // Innerer Loop: nur noch ein Add pro Pixel für cx
        for (int x = 0; x < width; ++x, cx += cx_step) {
            row[x] = mandelbrotIter(cx, cy, maxIter, bailoutSq);
        }
    }

    const auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace Nacktmull
