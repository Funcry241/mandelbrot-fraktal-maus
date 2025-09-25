///// Otter: Tile-Index -> Pixelzentrum; robust geklemmt, schnelle Inline-Helfer.
///// Schneefuchs: noexcept, deterministisch; Assertions nur in Debug; ASCII-only.
///// Maus: Keine Logs; Header/Source synchron; API stabil, [[nodiscard]] für Nutzungssicherheit.
///// Datei: src/heatmap_utils.hpp

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <utility>   // std::pair
#include <cassert>

namespace HeatmapUtils {

// -------------------------------------------------------------------------------------
// Sichtbare Host-Arrays (Entropy/Contrast) sicherstellen. Falls leer/inkonsistent,
// wird ein sichtbares Fallback-Muster (Gradient + Checker) erzeugt.
// -------------------------------------------------------------------------------------
inline void ensureHostArrays(std::vector<float>& entropy,
                             std::vector<float>& contrast,
                             int width, int height, int tilePx) noexcept
{
    const int px = std::max(1, tilePx);
    const int tx = (width  + px - 1) / px;
    const int ty = (height + px - 1) / px;
    const size_t N = static_cast<size_t>(tx) * static_cast<size_t>(ty);

    const bool needEnt = entropy.size()  != N || entropy.empty();
    const bool needCon = contrast.size() != N || contrast.empty();
    if (!needEnt && !needCon) return;

    if (needEnt) entropy.assign(N, 0.0f);
    if (needCon) contrast.assign(N, 0.0f);

    for (int y = 0; y < ty; ++y) {
        for (int x = 0; x < tx; ++x) {
            const size_t i = static_cast<size_t>(y) * tx + x;
            const float fx = (tx > 1) ? float(x) / float(tx - 1) : 0.0f;
            const float fy = (ty > 1) ? float(y) / float(ty - 1) : 0.0f;
            const float r  = std::min(1.0f, std::sqrt(fx*fx + fy*fy));
            entropy[i]  = 0.15f + 0.8f * r;

            const int  checker = ((x ^ y) & 1);
            const float mix    = 0.3f + 0.7f * ((fx + (1.0f - fy)) * 0.5f);
            contrast[i] = checker ? mix : (1.0f - mix);
        }
    }
}

// -------------------------------------------------------------------------------------
// Iterationspuffer -> Tile-Metriken.
// Entropy: binäre Entropie um den lokalen Mittelwert (0..1).
// Contrast: normierte Standardabweichung (leicht gestreckt).
// -------------------------------------------------------------------------------------
inline void computeTileStatsFromIterations(const std::vector<uint16_t>& iters,
                                           int W, int H, int tilePx, int maxIt,
                                           std::vector<float>& outEntropy,
                                           std::vector<float>& outContrast) noexcept
{
    if (W <= 0 || H <= 0 || tilePx <= 0 ||
        iters.size() != static_cast<size_t>(W) * static_cast<size_t>(H)) {
        outEntropy.clear();
        outContrast.clear();
        return;
    }

    const int px = std::max(1, tilePx);
    const int tx = (W + px - 1) / px;
    const int ty = (H + px - 1) / px;
    const size_t Ntiles = static_cast<size_t>(tx) * static_cast<size_t>(ty);

    outEntropy.assign(Ntiles, 0.f);
    outContrast.assign(Ntiles, 0.f);

    const float norm = (maxIt > 0) ? (1.f / float(maxIt)) : (1.f / 65535.f);

    auto clamp01  = [](float x) noexcept { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); };
    auto safe_log2= [](float p) noexcept { return (p > 0.f) ? std::log2(p) : 0.f; };

    for (int tyi = 0; tyi < ty; ++tyi) {
        for (int txi = 0; txi < tx; ++txi) {
            const int x0 = txi * px, y0 = tyi * px;
            const int x1 = std::min(W, x0 + px);
            const int y1 = std::min(H, y0 + px);
            const size_t idx = static_cast<size_t>(tyi) * tx + txi;

            // Mittelwert & Varianz
            double sum = 0.0, sum2 = 0.0; int n = 0;
            for (int y = y0; y < y1; ++y) {
                const uint16_t* row = &iters[static_cast<size_t>(y) * W];
                for (int x = x0; x < x1; ++x) {
                    const float v = float(row[x]) * norm;
                    sum += v; sum2 += double(v) * v; ++n;
                }
            }
            if (n <= 0) { outEntropy[idx] = 0.f; outContrast[idx] = 0.f; continue; }

            const float mean  = float(sum / n);
            const float var   = float(std::max(0.0, sum2 / n - double(mean) * mean));
            const float stdev = std::sqrt(var);

            // binäre Entropie (Verhältnis < mean)
            int below = 0;
            for (int y = y0; y < y1; ++y) {
                const uint16_t* row = &iters[static_cast<size_t>(y) * W];
                for (int x = x0; x < x1; ++x) {
                    below += (float(row[x]) * norm < mean);
                }
            }
            const float p = clamp01(float(below) / float(n));
            const float Hbin = clamp01(-(p * safe_log2(p) + (1.f - p) * safe_log2(1.f - p)));

            outEntropy[idx]  = Hbin;
            outContrast[idx] = clamp01(stdev * 3.0f); // leichte Streckung für Sichtbarkeit
        }
    }
}

// -------------------------------------------------------------------------------------
// Tile-Index -> Pixelzentrum (in ganzen Bildkoordinaten), robust geklemmt.
// [[nodiscard]] und noexcept für saubere Call-Sites.
// -------------------------------------------------------------------------------------
[[nodiscard]] inline std::pair<double,double> tileIndexToPixelCenter(
    int tileIndex,
    int tilesX, int tilesY,
    int width, int height) noexcept
{
    assert(tilesX > 0 && tilesY > 0 && "tiles must be > 0");
    assert(width  > 0 && height > 0    && "image size must be > 0");

    // In Release klemmen wir hart, um UB zu vermeiden.
    const int total = tilesX * tilesY;
    if (tileIndex < 0)       tileIndex = 0;
    if (tileIndex >= total)  tileIndex = total - 1;

    const int tileX = tileIndex % tilesX;
    const int tileY = tileIndex / tilesX;

    const double tileW = static_cast<double>(width)  / static_cast<double>(tilesX);
    const double tileH = static_cast<double>(height) / static_cast<double>(tilesY);

    const double px = (static_cast<double>(tileX) + 0.5) * tileW;
    const double py = (static_cast<double>(tileY) + 0.5) * tileH;

    return {px, py};
}

} // namespace HeatmapUtils
