///// Otter: Feature packer for AOP policy (dry-run) — E/C → tensor; lightweight, deterministic
///// Schneefuchs: Bounds-safe copy; zero-fill remainder; concise ASCII meta-log with ranges
///// Maus: Derives Tx×Ty from width/height/statsPx; uses NCHW [1,2,Ty,Tx]; no ORT dependency
///// Datei: src/ai/feature_packer.cpp
#include "feature_packer.hpp"
#include "luchs_log_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Repl { namespace Feat {

static inline int div_ceil(int a, int b) {
    return (a + (b - 1)) / b;
}

PackedFeatures pack_heatmap_features(const std::vector<float>& entropy,
                                     const std::vector<float>& contrast,
                                     int width, int height, int statsPx)
{
    const int px     = std::max(1, statsPx);
    const int Tx     = std::max(1, div_ceil(width,  px));
    const int Ty     = std::max(1, div_ceil(height, px));
    const size_t exp = static_cast<size_t>(Tx) * static_cast<size_t>(Ty);

    const size_t nE  = entropy.size();
    const size_t nC  = contrast.size();
    const size_t use = std::min(exp, std::min(nE, nC));

    PackedFeatures out;
    out.N = 1;
    out.C = 2;
    out.H = Ty;
    out.W = Tx;
    out.tilesX = Tx;
    out.tilesY = Ty;
    out.statsPx = px;
    out.data.assign(static_cast<size_t>(out.N) * out.C * out.H * out.W, 0.0f);

    // Fill channels: 0 = Entropy, 1 = Contrast
    // idx(ch, y, x) = ((ch * Ty) + y) * Tx + x
    auto idx = [Tx, Ty](int ch, int y, int x) -> size_t {
        return static_cast<size_t>((ch * Ty + y) * Tx + x);
    };

    for (size_t i = 0; i < use; ++i) {
        const int y = static_cast<int>(i / Tx);
        const int x = static_cast<int>(i % Tx);
        out.data[idx(0, y, x)] = entropy[i];
        out.data[idx(1, y, x)] = contrast[i];
    }

    // Meta ranges on used slices (avoid empty iterator UB)
    float eMin = 0.f, eMax = 0.f, cMin = 0.f, cMax = 0.f;
    if (use > 0) {
        auto mmE = std::minmax_element(entropy.begin(), entropy.begin() + static_cast<std::ptrdiff_t>(use));
        auto mmC = std::minmax_element(contrast.begin(), contrast.begin() + static_cast<std::ptrdiff_t>(use));
        eMin = *mmE.first; eMax = *mmE.second;
        cMin = *mmC.first; cMax = *mmC.second;
    }

    // Short ASCII meta-log for diagnostics
    LUCHS_LOG_HOST("[REPL/FEAT] tiles=%dx%d statsPx=%d N=%zu used=%zu E[min=%.4f max=%.4f] C[min=%.4f max=%.4f]",
                   Tx, Ty, px, exp, use, eMin, eMax, cMin, cMax);

    if (use < exp) {
        // Inform about zero-fill remainder without spamming
        const size_t rem = exp - use;
        if (rem > 0) {
            LUCHS_LOG_HOST("[REPL/FEAT] zero-filled=%zu (entropy=%zu contrast=%zu expected=%zu)",
                           rem, nE, nC, exp);
        }
    }

    return out;
}

}} // namespace Repl::Feat
