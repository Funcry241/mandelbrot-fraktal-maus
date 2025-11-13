///// Otter: Feature packer - Bandit-Featurematrix (~20D); deterministisch, robust gegenüber Rändern.
///// Schneefuchs: 3x3-Stats, Grad-Proxy, NDC-Koords, r/theta, center-bias; ASCII-Metalog; keine Fremdlibs.
///// Maus: Bandit-Matrix row-major [Tiles x d]; stride == dim; Shapes exakt geloggt.
/// // Datei: src/ai/feature_packer.cpp
#include "feature_packer.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#include <algorithm>
#include <cmath>

namespace Repl { namespace Feat {

static inline int div_ceil(int a, int b) { return (a + (b - 1)) / b; }

// -------------------- Bandit-Featurematrix ----------------------------------
// Feature-Layout je Tile (dim = 20):
// 0: E, 1: C,
// 2: meanE3, 3: meanC3,
// 4: varE3,  5: varC3,
// 6: gradE,  7: gradC,
// 8: rangeE, 9: rangeC,            // max-min im 3x3
// 10: x_ndc, 11: y_ndc,            // in [-1,1], Tile-Center
// 12: r, 13: cosTheta, 14: sinTheta, // theta = atan2(y_ndc, x_ndc)
// 15: centerBias,                  // exp(-r^2 / sigma^2), sigma = Settings::TargetBias::sigmaNdc
// 16: eRel = E - meanE_all,        // global zentriert
// 17: cRel = C - meanC_all,
// 18: eStd = (stdE_all>0)?(E-meanE)/stdE:0
// 19: cStd = (stdC_all>0)?(C-meanC)/stdC:0
static inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

BanditMatrix make_bandit_feature_matrix(const std::vector<float>& entropy,
                                        const std::vector<float>& contrast,
                                        int width, int height, int statsPx)
{
    const int px     = std::max(1, statsPx);
    const int Tx     = std::max(1, div_ceil(width,  px));
    const int Ty     = std::max(1, div_ceil(height, px));
    const size_t Ntx = static_cast<size_t>(Tx) * static_cast<size_t>(Ty);

    const size_t nE  = entropy.size();
    const size_t nC  = contrast.size();
    const size_t use = std::min(Ntx, std::min(nE, nC));

    // Globale Mittel/Std für zentrierte Features
    double sumE = 0.0, sumE2 = 0.0, sumC = 0.0, sumC2 = 0.0;
    for (size_t i = 0; i < use; ++i) {
        const double e = static_cast<double>(entropy[i]);
        const double c = static_cast<double>(contrast[i]);
        sumE  += e;
        sumE2 += e * e;
        sumC  += c;
        sumC2 += c * c;
    }
    const double invN  = (use > 0) ? (1.0 / static_cast<double>(use)) : 0.0;
    const double meanE = sumE * invN;
    const double meanC = sumC * invN;
    const double varE  = std::max(0.0, sumE2 * invN - meanE * meanE);
    const double varC  = std::max(0.0, sumC2 * invN - meanC * meanC);
    const double stdE  = std::sqrt(varE);
    const double stdC  = std::sqrt(varC);

    BanditMatrix out;
    out.dim     = 20;
    out.tilesX  = Tx;
    out.tilesY  = Ty;
    out.stride  = out.dim;
    out.statsPx = px;
    out.data.assign(Ntx * static_cast<size_t>(out.dim), 0.0f);

    auto idxEC = [Tx](int y, int x) -> size_t {
        return static_cast<size_t>(y) * static_cast<size_t>(Tx) + static_cast<size_t>(x);
    };

    auto getE = [&](int y, int x) -> float {
        const int yy = (y < 0) ? 0 : ((y >= Ty) ? (Ty - 1) : y);
        const int xx = (x < 0) ? 0 : ((x >= Tx) ? (Tx - 1) : x);
        const size_t i = idxEC(yy, xx);
        return (i < entropy.size()) ? entropy[i] : 0.0f;
    };
    auto getC = [&](int y, int x) -> float {
        const int yy = (y < 0) ? 0 : ((y >= Ty) ? (Ty - 1) : y);
        const int xx = (x < 0) ? 0 : ((x >= Tx) ? (Tx - 1) : x);
        const size_t i = idxEC(yy, xx);
        return (i < contrast.size()) ? contrast[i] : 0.0f;
    };

    const double sigma = (Settings::TargetBias::sigmaNdc > 0.0)
                       ? Settings::TargetBias::sigmaNdc
                       : 0.65;

    for (int y = 0; y < Ty; ++y) {
        for (int x = 0; x < Tx; ++x) {
            const size_t tileIdx = idxEC(y, x);
            float* row = out.data.data() + tileIdx * static_cast<size_t>(out.dim);

            const float E = (tileIdx < entropy.size())  ? entropy[tileIdx]  : 0.0f;
            const float C = (tileIdx < contrast.size()) ? contrast[tileIdx] : 0.0f;

            // 3x3 Stats um die aktuelle Kachel
            double sum_e  = 0.0;
            double sum_c  = 0.0;
            double sum_e2 = 0.0;
            double sum_c2 = 0.0;
            float eMin = E, eMax = E, cMin = C, cMax = C;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const float ev = getE(y + dy, x + dx);
                    const float cv = getC(y + dy, x + dx);
                    sum_e  += ev;
                    sum_c  += cv;
                    sum_e2 += static_cast<double>(ev) * static_cast<double>(ev);
                    sum_c2 += static_cast<double>(cv) * static_cast<double>(cv);
                    eMin = std::min(eMin, ev);
                    eMax = std::max(eMax, ev);
                    cMin = std::min(cMin, cv);
                    cMax = std::max(cMax, cv);
                }
            }

            const double cnt    = 9.0;
            const double invCnt = 1.0 / cnt;
            const double mean_e = sum_e * invCnt;
            const double mean_c = sum_c * invCnt;

            const float meanE3 = static_cast<float>(mean_e);
            const float meanC3 = static_cast<float>(mean_c);
            const float varE3  = static_cast<float>(std::max(0.0, sum_e2 * invCnt - mean_e * mean_e));
            const float varC3  = static_cast<float>(std::max(0.0, sum_c2 * invCnt - mean_c * mean_c));
            const float rangeE = eMax - eMin;
            const float rangeC = cMax - cMin;

            // Gradient-Proxy (vorwärts-Differenzen, randgeklemmt)
            const float gradE = std::fabs(getE(y, x + 1) - E) + std::fabs(getE(y + 1, x) - E);
            const float gradC = std::fabs(getC(y, x + 1) - C) + std::fabs(getC(y + 1, x) - C);

            // NDC-Koords (Tile-Center relativ zur Tile-Anzahl)
            const float ndcX = clamp01((static_cast<float>(x) + 0.5f) / static_cast<float>(Tx)) * 2.0f - 1.0f;
            const float ndcY = clamp01((static_cast<float>(y) + 0.5f) / static_cast<float>(Ty)) * 2.0f - 1.0f;
            const float r2   = ndcX * ndcX + ndcY * ndcY;
            const float r    = std::sqrt(r2);
            const float ang  = std::atan2(ndcY, ndcX);
            const float cb   = std::exp(-(r2) / static_cast<float>(sigma * sigma));

            const float eRel = static_cast<float>(static_cast<double>(E) - meanE);
            const float cRel = static_cast<float>(static_cast<double>(C) - meanC);
            const float eStd = (stdE > 1e-12)
                ? static_cast<float>((static_cast<double>(E) - meanE) / stdE)
                : 0.0f;
            const float cStd = (stdC > 1e-12)
                ? static_cast<float>((static_cast<double>(C) - meanC) / stdC)
                : 0.0f;

            // Write row
            row[0]  = E;
            row[1]  = C;
            row[2]  = meanE3;
            row[3]  = meanC3;
            row[4]  = varE3;
            row[5]  = varC3;
            row[6]  = gradE;
            row[7]  = gradC;
            row[8]  = rangeE;
            row[9]  = rangeC;
            row[10] = ndcX;
            row[11] = ndcY;
            row[12] = r;
            row[13] = std::cos(ang);
            row[14] = std::sin(ang);
            row[15] = cb;
            row[16] = eRel;
            row[17] = cRel;
            row[18] = eStd;
            row[19] = cStd;
        }
    }

    LUCHS_LOG_HOST("[REPL/FEAT] bandit-matrix tiles=%dx%d d=%d statsPx=%d rows=%zu stride=%d",
                   Tx, Ty, out.dim, px, Ntx, out.stride);
    return out;
}

}} // namespace Repl::Feat
