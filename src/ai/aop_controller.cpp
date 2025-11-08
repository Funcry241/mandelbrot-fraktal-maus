///// Otter: AOP controller — centralizes [REPL/POLICY] logging; Phase-1 stub + dry-run top3 + delta to overlay
///// Schneefuchs: /WX-safe; ASCII-only; compile-time gates via Settings::Ai; no side-effects
///// Maus: Cadence = Settings::PerfLog (warmup + everyN); ep from Settings::Ai::ep; stats grid = Kolibri::desiredTilePx
///// Datei: src/ai/aop_controller.cpp

#include "pch.hpp"
#include "ai/aop_controller.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "renderer_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>

// ----- Phase-1 Telemetry (externally readable by HUD) ------------------------
namespace AOP_Telemetry {
    // Overlap preview in NDC (for overlay alignment tests)
    // (Exported via extern elsewhere if needed; keep static here for stub)
    static float g_ai_ndc_x  = 0.0f;
    static float g_ai_ndc_y  = 0.0f;
    static int   g_ai_valid  = 0;

    // Overlay-mark position (mini-overlay space)
    static float g_ai_ndc_ovl_x = 0.0f;
    static float g_ai_ndc_ovl_y = 0.0f;
    static int   g_ai_ov_valid  = 0;

    // Frame counter for correlation
    static unsigned long long g_ai_frame_id = 0ULL;
} // namespace AOP_Telemetry

namespace Repl { namespace Policy {

// Helper: clamp
static inline float _clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Helper: safe sqrt
static inline float _safe_len(float dx, float dy) {
    const float s = dx*dx + dy*dy;
    return s > 0.0f ? std::sqrt(s) : 0.0f;
}

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state) {
    (void)fctx;
    Decision d{};

    // Nur loggen, wenn global aktiviert und die Perf-Cadence greift.
    if constexpr (Settings::Ai::enabled && Settings::Ai::aopEnabled
                  && Settings::performanceLogging && Settings::PerfLog::enabled) {

        const int warm = Settings::PerfLog::warmupFrames;
        const int step = Settings::PerfLog::everyN;

        if (state.frameCount > warm && (state.frameCount % step) == 0) {
            // --- Dry-run: Top-3 Tiles basierend auf state.h_entropy/h_contrast ---
            const int statsPx = std::max(1, Settings::Kolibri::desiredTilePx);
            const int tilesX  = (state.width  + statsPx - 1) / statsPx;
            const int tilesY  = (state.height + statsPx - 1) / statsPx;

            const std::size_t nGrid = (tilesX > 0 && tilesY > 0)
                                      ? static_cast<std::size_t>(tilesX) * static_cast<std::size_t>(tilesY)
                                      : 0;

            const float* E = state.h_entropy.data();
            const float* C = state.h_contrast.data();
            const std::size_t nE = state.h_entropy.size();
            const std::size_t nC = state.h_contrast.size();

            const std::size_t N  = std::min(nGrid, std::min(nE, nC));

            if (N == 0 || tilesX <= 0 || tilesY <= 0) {
                LUCHS_LOG_HOST("[REPL/POLICY] evaluate (stub, ep=%s) dry-run: no-metrics N=%zu statsPx=%d tiles=%dx%d",
                               Settings::Ai::ep, N, statsPx, tilesX, tilesY);
                return d;
            }

            // Score = simple mean normalization + max-pick (stub)
            struct Scored { std::size_t idx; float s; };
            Scored top[3] = { {0, -1e9f},{0, -1e9f},{0, -1e9f} };

            // Precompute simple stats for normalization
            double sumE = 0.0, sumE2 = 0.0, sumC = 0.0, sumC2 = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const float e = E[i], c = C[i];
                sumE  += e; sumE2 += (double)e*e;
                sumC  += c; sumC2 += (double)c*c;
            }
            const double invN = (N > 0) ? (1.0 / (double)N) : 0.0;
            const double meanE = sumE * invN;
            const double meanC = sumC * invN;
            const double varE  = std::max(0.0, sumE2 * invN - meanE * meanE);
            const double varC  = std::max(0.0, sumC2 * invN - meanC * meanC);
            const double stdE  = std::sqrt(varE);
            const double stdC  = std::sqrt(varC);

            // Simple z-score combo: s = zE + zC
            for (std::size_t i = 0; i < N; ++i) {
                const float e = E[i], c = C[i];
                const float zE = (float)((stdE > 1e-12) ? ((e - (float)meanE) / (float)stdE) : 0.0f);
                const float zC = (float)((stdC > 1e-12) ? ((c - (float)meanC) / (float)stdC) : 0.0f);
                const float s  = zE + zC;

                // keep 3 best
                if (s > top[0].s) {
                    top[2] = top[1]; top[1] = top[0]; top[0] = { i, s };
                } else if (s > top[1].s) {
                    top[2] = top[1]; top[1] = { i, s };
                } else if (s > top[2].s) {
                    top[2] = { i, s };
                }
            }

            // Approximate NDC center for best tile
            const std::size_t best = top[0].idx;
            const int tx = (int)(best % (std::size_t)tilesX);
            const int ty = (int)(best / (std::size_t)tilesX);

            const float pxCenterX = (tx + 0.5f) * (float)statsPx;
            const float pxCenterY = (ty + 0.5f) * (float)statsPx;

            const float ndcX = _clamp((pxCenterX / (float)state.width)  * 2.0f - 1.0f, -1.0f, 1.0f);
            const float ndcY = _clamp((pxCenterY / (float)state.height) * 2.0f - 1.0f, -1.0f, 1.0f);

            // Overlay preview position (optional)
            const float ovX = ndcX * 0.75f;
            const float ovY = ndcY * 0.75f;
            const int   ovVal = 1;

            // Delta to current interest/crosshair if available
            const float dx = ndcX - state.interest.ndcX;
            const float dy = ndcY - state.interest.ndcY;
            const float delta = state.interest.valid ? _safe_len(dx, dy) : -1.0f;

            LUCHS_LOG_HOST("[REPL/POLICY] dry-run ep=%s tiles=%dx%d statsPx=%d best=%zu s=%.3f ndc=(%.3f,%.3f) delta=%.3f",
                           Settings::Ai::ep, tilesX, tilesY, statsPx, best, top[0].s, ndcX, ndcY, delta);

            // export telemetry (HUD may read)
            AOP_Telemetry::g_ai_ndc_x = ndcX;
            AOP_Telemetry::g_ai_ndc_y = ndcY;
            AOP_Telemetry::g_ai_valid = 1;

            AOP_Telemetry::g_ai_ndc_ovl_x = ovX;
            AOP_Telemetry::g_ai_ndc_ovl_y = ovY;
            AOP_Telemetry::g_ai_ov_valid  = ovVal;

            ++AOP_Telemetry::g_ai_frame_id;

            // Stub decision (no effect)
            d.order = 1;
            d.rebase = false;
            d.enablePerturb = false;
        }
    }

    return d;
}

}} // namespace Repl::Policy
