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
    // Last policy/overlay comparison
    float g_ai_last_delta   = -1.0f;
    float g_ai_ndc_pol_x    = 0.0f;
    float g_ai_ndc_pol_y    = 0.0f;
    float g_ai_ndc_ovl_x    = 0.0f;
    float g_ai_ndc_ovl_y    = 0.0f;
    int   g_ai_ov_valid     = 0;
    unsigned long long g_ai_frame_id = 0ULL;
} // namespace AOP_Telemetry

namespace Repl { namespace Policy {

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

            const std::size_t nE = state.h_entropy.size();
            const std::size_t nC = state.h_contrast.size();
            const std::size_t N  = std::min(nGrid, std::min(nE, nC));

            if (N == 0 || tilesX <= 0 || tilesY <= 0) {
                LUCHS_LOG_HOST("[REPL/POLICY] evaluate (stub, ep=%s) dry-run: no-metrics N=%zu statsPx=%d tiles=%dx%d",
                               Settings::Ai::ep, N, statsPx, tilesX, tilesY);
            } else {
                // Gewichte (einfaches konvexes Kombi-Signal)
                const float wE = 0.60f;
                const float wC = 0.40f;

                // Top-3 ohne Allokation
                struct Top { int idx; float score; int tx; int ty; };
                Top t0{-1, -1e30f, -1, -1}, t1{-1, -1e30f, -1, -1}, t2{-1, -1e30f, -1, -1};

                for (std::size_t i = 0; i < N; ++i) {
                    const float e = state.h_entropy[i];
                    const float c = state.h_contrast[i];
                    const float s = wE * e + wC * c;

                    if (s > t0.score) { t2 = t1; t1 = t0; t0 = { static_cast<int>(i), s, 0, 0 }; }
                    else if (s > t1.score) { t2 = t1; t1 = { static_cast<int>(i), s, 0, 0 }; }
                    else if (s > t2.score) { t2 = { static_cast<int>(i), s, 0, 0 }; }
                }

                auto idxToXY = [&](int idx, int& x, int& y) {
                    if (idx < 0) { x = -1; y = -1; return; }
                    x = (tilesX > 0) ? (idx % tilesX) : -1;
                    y = (tilesX > 0) ? (idx / tilesX) : -1;
                };
                idxToXY(t0.idx, t0.tx, t0.ty);
                idxToXY(t1.idx, t1.tx, t1.ty);
                idxToXY(t2.idx, t2.tx, t2.ty);

                // NDC-Zentrum des „chosen“ Tiles (nur für Logs)
                float ndcX = 0.0f, ndcY = 0.0f;
                if (t0.tx >= 0 && t0.ty >= 0) {
                    const float px = (static_cast<float>(t0.tx) + 0.5f) * static_cast<float>(statsPx);
                    const float py = (static_cast<float>(t0.ty) + 0.5f) * static_cast<float>(statsPx);
                    ndcX = (state.width  > 0) ? (px / static_cast<float>(state.width))  * 2.0f - 1.0f : 0.0f;
                    ndcY = (state.height > 0) ? 1.0f - (py / static_cast<float>(state.height)) * 2.0f : 0.0f;
                }

                // Hauptzeile: Top-3 + chosen + NDC
                LUCHS_LOG_HOST(
                    "[REPL/POLICY] evaluate (stub, ep=%s) dry-run: N=%zu statsPx=%d tiles=%dx%d "
                    "top3={%d(%d,%d):%.4f | %d(%d,%d):%.4f | %d(%d,%d):%.4f} "
                    "chosen=%d(%d,%d) ndc=(%.3f,%.3f)",
                    Settings::Ai::ep, N, statsPx, tilesX, tilesY,
                    t0.idx, t0.tx, t0.ty, t0.score,
                    t1.idx, t1.tx, t1.ty, t1.score,
                    t2.idx, t2.tx, t2.ty, t2.score,
                    t0.idx, t0.tx, t0.ty, ndcX, ndcY
                );

                // NEU (Phase-1 Telemetrie): Delta zwischen Policy-NDC und Overlay-Interest-NDC
                {
                    const int ovValid = state.interest.valid ? 1 : 0;
                    float ovX = 0.0f, ovY = 0.0f;
                    if (state.interest.valid) {
                        ovX = static_cast<float>(state.interest.ndcX);
                        ovY = static_cast<float>(state.interest.ndcY);
                    }
                    const float dx = ndcX - ovX;
                    const float dy = ndcY - ovY;
                    const float delta = state.interest.valid ? std::sqrt(dx*dx + dy*dy) : -1.0f;

                    LUCHS_LOG_HOST(
                        "[REPL/POLICY] delta=%.4f ndc_pol=(%.3f,%.3f) ndc_ovl=(%.3f,%.3f) ov_valid=%d",
                        delta, ndcX, ndcY, ovX, ovY, ovValid
                    );

                    // Write HUD-readable telemetry
                    AOP_Telemetry::g_ai_last_delta = delta;
                    AOP_Telemetry::g_ai_ndc_pol_x  = ndcX;
                    AOP_Telemetry::g_ai_ndc_pol_y  = ndcY;
                    AOP_Telemetry::g_ai_ndc_ovl_x  = ovX;
                    AOP_Telemetry::g_ai_ndc_ovl_y  = ovY;
                    AOP_Telemetry::g_ai_ov_valid   = ovValid;
                    AOP_Telemetry::g_ai_frame_id   = static_cast<unsigned long long>(state.frameCount);
                }
            }
        }
    }

    // Phase-1: keinerlei Änderungen am Verhalten
    return d;
}

}} // namespace Repl::Policy
