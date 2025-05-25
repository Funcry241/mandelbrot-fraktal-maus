// src/compute/boundary_gpu.cpp

#include "mandelbrot.hpp"            // Deklariert computeBoundaryGPU-Signatur
#include <vector>
#include <cmath>
#include <limits>
#include <boost/multiprecision/cpp_dec_float.hpp>

// Struktur für interne Punkte
struct C { float x, y, g; };

std::pair<float,float> computeBoundaryGPU(
    double z, double ox, double oy,
    int w, int h, int sampleStep, int maxIter
) {
    int baseStep = std::max(1, sampleStep);
    int step     = std::max(1, int(baseStep * std::sqrt(z) * 0.5));

    std::vector<C> pts;
    pts.reserve((w/step)*(h/step));
    float maxG = 0.0f;

    auto norm = [&](int px, int py) {
        float fx = ((px / float(w)  - 0.5f) * 3.5f) / float(z) + float(ox);
        float fy = ((0.5f - py / float(h))   * 2.0f) / float(z) + float(oy);
        float a = 0.0f, b = 0.0f;
        int   i = 0;
        while (a*a + b*b <= 4.0f && i < maxIter) {
            float na = a*a - b*b + fx;
            float nb = 2.0f*a*b + fy;
            a = na; b = nb;
            ++i;
        }
        float smooth = 0.0f;
        float mag2   = a*a + b*b + 1e-6f;
        if (mag2 > 1.0f) {
            float l1 = log2f(mag2);
            if (l1 > 0.0f) smooth = log2f(l1);
        }
        return i + 1 - smooth;
    };

    for (int y = step; y < h - step; y += step) {
        for (int x = step; x < w - step; x += step) {
            float c  = norm(x, y);
            float gx = norm(x + step, y);
            float gy = norm(x, y + step);
            float g  = fabsf(gx - c) + fabsf(gy - c);
            float fx = ((x / float(w)  - 0.5f) * 3.5f) / float(z) + float(ox);
            float fy = ((0.5f - y / float(h))   * 2.0f) / float(z) + float(oy);
            pts.push_back({ fx, fy, g });
            if (g > maxG) maxG = g;
        }
    }

    float cutoff = std::max(0.0f, maxG * 0.5f);
    std::vector<const C*> focus;
    focus.reserve(pts.size());
    for (auto& p : pts)
        if (p.g >= cutoff) focus.push_back(&p);
    if (focus.empty())
        for (auto& p : pts) focus.push_back(&p);

    const C* best = focus.front();
    float bestD   = std::numeric_limits<float>::max();
    for (auto p : focus) {
        float dx = p->x - float(ox);
        float dy = p->y - float(oy);
        float d2 = dx*dx + dy*dy;
        if (d2 < bestD) {
            bestD = d2;
            best  = p;
        }
    }
    return { best->x, best->y };
}
