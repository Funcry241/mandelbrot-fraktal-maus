///// Otter: AOP controller — centralizes [REPL/POLICY] logging (shadow only) + feature packing (no ORT)
///// Schneefuchs: /WX-safe; ASCII-only; if constexpr gates; uses AOP_Telemetry; statsPx from FrameContext
///// Maus: Size-safe math; stabile One-Liner; dry-run only; Kolibri desiredTilePx fallback
///// Datei: src/ai/aop_controller.cpp

#include "pch.hpp"
#include "ai/aop_controller.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "renderer_state.hpp"
#include "frame_context.hpp"     // vollständige Definition von FrameContext nötig
#include "ai/aop_telemetry.hpp"
#include "ai/feature_packer.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cfloat>

namespace Repl { namespace Policy {

static inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state)
{
    Decision d{};

    // Compile-time/logging cadence gate (shadow/preview frequency).
    if constexpr (!(Settings::performanceLogging && Settings::PerfLog::enabled)) {
        return d;
    }

    // Runtime cadence (tie preview to PERF cadence)
    const int warmupFrames = Settings::PerfLog::warmupFrames;
    const int everyN       = (Settings::PerfLog::everyN > 0) ? Settings::PerfLog::everyN : 1;
    if (!(state.frameCount > warmupFrames && (state.frameCount % everyN) == 0)) {
        return d;
    }

    // Grid: bevorzugt fctx.statsTileSize (post-metrics), sonst desiredTilePx
    int statsPx_i = fctx.statsTileSize;
    if (statsPx_i <= 0) statsPx_i = Settings::Kolibri::desiredTilePx;  // konstante muss initialisiert sein (MSVC)
    const int px_i = std::max(1, statsPx_i);

    if (state.width <= 0 || state.height <= 0) {
        LUCHS_LOG_HOST("[REPL/POLICY] dry-run: invalid dims w=%d h=%d", state.width, state.height);
        return d;
    }

    const size_t px = static_cast<size_t>(px_i);
    const size_t w  = static_cast<size_t>(state.width);
    const size_t h  = static_cast<size_t>(state.height);

    const size_t tilesX = (w + px - 1) / px;
    const size_t tilesY = (h + px - 1) / px;
    const size_t nGrid  = tilesX * tilesY;

    // Metrics
    const float* E = state.h_entropy.data();
    const float* C = state.h_contrast.data();
    const size_t nE = state.h_entropy.size();
    const size_t nC = state.h_contrast.size();
    const size_t N  = std::min(nGrid, std::min(nE, nC));

    if (N == 0 || tilesX == 0 || tilesY == 0 || E == nullptr || C == nullptr) {
        const unsigned long long uu_tilesX = static_cast<unsigned long long>(tilesX);
        const unsigned long long uu_tilesY = static_cast<unsigned long long>(tilesY);
        const unsigned long long uu_px     = static_cast<unsigned long long>(px);
        const unsigned long long uu_N      = static_cast<unsigned long long>(N);
        LUCHS_LOG_HOST("[REPL/POLICY] dry-run: no-metrics N=%llu tiles=%llux%llu statsPx=%llu",
                       uu_N, uu_tilesX, uu_tilesY, uu_px);
        return d;
    }

    // ---- Feature packing (logs [REPL/FEAT] intern) -------------------------
    {
        using namespace Repl::Feat;
        (void)pack_heatmap_features(state.h_entropy, state.h_contrast,
                                    static_cast<int>(w), static_cast<int>(h), px_i);
    }

    // -------------------- Simple z-score scoring (preview) -------------------
    struct Scored { size_t idx; float s; };
    Scored top[3] = { {0, -FLT_MAX}, {0, -FLT_MAX}, {0, -FLT_MAX} };

    double sumE = 0.0, sumE2 = 0.0, sumC = 0.0, sumC2 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const float e = E[i], c = C[i];
        sumE  += static_cast<double>(e); sumE2 += static_cast<double>(e) * static_cast<double>(e);
        sumC  += static_cast<double>(c); sumC2 += static_cast<double>(c) * static_cast<double>(c);
    }
    const double invN = 1.0 / static_cast<double>(N);
    const double meanE = sumE * invN;
    const double meanC = sumC * invN;
    const double varE  = std::max(0.0, sumE2 * invN - meanE * meanE);
    const double varC  = std::max(0.0, sumC2 * invN - meanC * meanC);
    const double stdE  = std::sqrt(varE);
    const double stdC  = std::sqrt(varC);

    for (size_t i = 0; i < N; ++i) {
        const float e = E[i], c = C[i];
        const float zE = (stdE > 1e-12) ? static_cast<float>((static_cast<double>(e) - meanE) / stdE) : 0.0f;
        const float zC = (stdC > 1e-12) ? static_cast<float>((static_cast<double>(c) - meanC) / stdC) : 0.0f;
        const float s  = zE + zC;

        if (s > top[0].s) {
            top[2] = top[1]; top[1] = top[0]; top[0] = { i, s };
        } else if (s > top[1].s) {
            top[2] = top[1]; top[1] = { i, s };
        } else if (s > top[2].s) {
            top[2] = { i, s };
        }
    }

    const size_t best = top[0].idx;
    const size_t tx = tilesX ? (best % tilesX) : 0;
    const size_t ty = tilesY ? (best / tilesX) : 0;

    const float pxCenterX = (static_cast<float>(tx) + 0.5f) * static_cast<float>(px_i);
    const float pxCenterY = (static_cast<float>(ty) + 0.5f) * static_cast<float>(px_i);

    const float ndcX = clamp01(pxCenterX / static_cast<float>(state.width))  * 2.0f - 1.0f;
    const float ndcY = clamp01(pxCenterY / static_cast<float>(state.height)) * 2.0f - 1.0f;

    // Telemetry export (extern)
    AOP_Telemetry::g_ai_ndc_pol_x = ndcX;
    AOP_Telemetry::g_ai_ndc_pol_y = ndcY;

    // Overlay-Preview: simple scaling
    AOP_Telemetry::g_ai_ndc_ovl_x = ndcX * 0.75f;
    AOP_Telemetry::g_ai_ndc_ovl_y = ndcY * 0.75f;
    AOP_Telemetry::g_ai_ov_valid  = 1;

    // Diagnose: NDC-Center (0,0) → Policy-NDC Distanz (L2)
    AOP_Telemetry::g_ai_last_delta = std::sqrt(ndcX * ndcX + ndcY * ndcY);

    // Monotone Frame-ID
    ++AOP_Telemetry::g_ai_frame_id;

    // Log
    const unsigned long long uu_tilesX = static_cast<unsigned long long>(tilesX);
    const unsigned long long uu_tilesY = static_cast<unsigned long long>(tilesY);
    const unsigned long long uu_px     = static_cast<unsigned long long>(px);
    const unsigned long long uu_best   = static_cast<unsigned long long>(best);

    LUCHS_LOG_HOST("[REPL/POLICY] dry-run tiles=%llux%llu statsPx=%llu best=%llu score=%.3f ndc=(%.3f,%.3f)",
                   uu_tilesX, uu_tilesY, uu_px, uu_best, top[0].s, ndcX, ndcY);

    // Stub decision (shadow-only)
    d.order = 1;
    d.rebase = false;
    d.enablePerturb = false;
    return d;
}

}} // namespace Repl::Policy
