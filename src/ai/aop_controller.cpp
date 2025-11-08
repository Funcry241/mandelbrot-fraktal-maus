///// Otter: AOP dry-run — top-k tiles & scores; zero side-effects; ASCII one-liner
///// Schneefuchs: No hidden macros; centralized [REPL/*] logs; guard empty metrics; stable cadence
///// Maus: Uses statsPx grid (Kolibri) and entropy/contrast to rank; returns unchanged Decision
///// Datei: src/ai/aop_controller.cpp

#include "pch.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>
#include <cmath>

#include "ai/aop_controller.hpp"   // Repl::Policy::Decision, signature
#include "frame_context.hpp"       // FrameContext (entropy/contrast/statsTileSize)
#include "renderer_resources.hpp"  // RendererState
#include "settings.hpp"
#include "luchs_log_host.hpp"

namespace Repl {
namespace Policy {

// --- helpers -----------------------------------------------------------------

struct TileRank {
    int   idx   = -1;
    int   tx    = 0;
    int   ty    = 0;
    float score = 0.0f;
};

// safe accessor with bounds check (returns 0 if OOB or empty)
static inline float get_or_zero(const std::vector<float>& v, std::size_t i) {
    return (i < v.size()) ? v[i] : 0.0f;
}

// convert tile index -> (x,y) tile coords
static inline void idx_to_xy(int idx, int tilesX, int& x, int& y) {
    if (idx < 0) { x = y = -1; return; }
    x = idx % tilesX; y = idx / tilesX;
}

// center of tile in NDC (-1..+1), based on pixel center
static inline void tile_center_ndc(int tx, int ty, int statsPx, int width, int height, float& ndcX, float& ndcY) {
    const float px = (static_cast<float>(tx) + 0.5f) * static_cast<float>(statsPx);
    const float py = (static_cast<float>(ty) + 0.5f) * static_cast<float>(statsPx);
    ndcX = (px / std::max(1, width))  * 2.0f - 1.0f;
    // NDC Y typically grows upward; screen Y grows downward → flip:
    ndcY = 1.0f - (py / std::max(1, height)) * 2.0f;
}

// --- policy (dry-run): rank tiles and log top-3 --------------------------------

Decision evaluate_tile_policy(FrameContext& fctx, RendererState& state)
{
    (void)state; // not used in dry-run

    Decision d{}; // NOTE: we do not change behavior in Phase-1 (dry-run only)

    if constexpr (!(Settings::Ai::enabled && Settings::Ai::aopEnabled)) {
        return d; // disabled → no-op
    }

    // choose stats grid in pixels
    const int statsPx = std::max(1, (fctx.statsTileSize > 0
                         ? fctx.statsTileSize
                         : (Settings::Kolibri::gridScreenConstant
                               ? Settings::Kolibri::desiredTilePx
                               : std::max(1, fctx.tileSize))));

    // derive grid size
    const int tilesX = (fctx.width  + statsPx - 1) / statsPx;
    const int tilesY = (fctx.height + statsPx - 1) / statsPx;
    const std::size_t N = static_cast<std::size_t>(std::max(0, tilesX)) *
                          static_cast<std::size_t>(std::max(0, tilesY));

    // guards
    if (N == 0 || fctx.entropy.empty() || fctx.contrast.empty()) {
        LUCHS_LOG_HOST("[REPL/POLICY] dry-run: no-metrics N=%zu statsPx=%d", N, statsPx);
        return d;
    }
    if (fctx.entropy.size() != N || fctx.contrast.size() != N) {
        LUCHS_LOG_HOST("[REPL/POLICY] dry-run: size-mismatch ent=%zu con=%zu N=%zu statsPx=%d",
                       fctx.entropy.size(), fctx.contrast.size(), N, statsPx);
        return d;
    }

    // weights: simple convex combo (may be tuned later)
    constexpr float wE = 0.60f; // entropy weight
    constexpr float wC = 0.40f; // contrast weight

    // rank top-3 in one pass (no allocations)
    TileRank top[3]; // sorted descending by score
    auto try_push = [&](int i, float s) {
        if (s <= top[2].score) return;
        // insert into top[0..2]
        if (s > top[0].score) {
            top[2] = top[1]; top[1] = top[0];
            top[0] = { i, 0, 0, s };
        } else if (s > top[1].score) {
            top[2] = top[1];
            top[1] = { i, 0, 0, s };
        } else {
            top[2] = { i, 0, 0, s };
        }
    };

    for (int y = 0, i = 0; y < tilesY; ++y) {
        for (int x = 0; x < tilesX; ++x, ++i) {
            const float e = get_or_zero(fctx.entropy,  static_cast<std::size_t>(i));
            const float c = get_or_zero(fctx.contrast, static_cast<std::size_t>(i));
            // simple score; later we may add center bias (already handled elsewhere)
            const float s = wE * e + wC * c;
            try_push(i, s);
        }
    }

    idx_to_xy(top[0].idx, tilesX, top[0].tx, top[0].ty);
    idx_to_xy(top[1].idx, tilesX, top[1].tx, top[1].ty);
    idx_to_xy(top[2].idx, tilesX, top[2].tx, top[2].ty);

    float ndcX0=0.f, ndcY0=0.f;
    tile_center_ndc(top[0].tx, top[0].ty, statsPx, fctx.width, fctx.height, ndcX0, ndcY0);

    // ASCII one-liner: top3 and chosen (dry-run, no behavior change)
    LUCHS_LOG_HOST(
        "[REPL/POLICY] dry-run: N=%zu statsPx=%d top3={%d(%d,%d):%.4f | %d(%d,%d):%.4f | %d(%d,%d):%.4f} "
        "chosen=%d(%d,%d) ndc=(%.3f,%.3f)",
        N, statsPx,
        top[0].idx, top[0].tx, top[0].ty, top[0].score,
        top[1].idx, top[1].tx, top[1].ty, top[1].score,
        top[2].idx, top[2].tx, top[2].ty, top[2].score,
        top[0].idx, top[0].tx, top[0].ty, ndcX0, ndcY0
    );

    // Phase-1: do NOT modify state or d (no side effects)
    return d;
}

} // namespace Policy
} // namespace Repl
