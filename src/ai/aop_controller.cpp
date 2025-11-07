///// Otter: AOP controller — centralizes [REPL/POLICY] logging; Phase-1 stub
///// Schneefuchs: /WX-safe; ASCII-only; compile-time gates via Settings::Ai
///// Maus: Cadence = Settings::PerfLog (warmup + everyN); ep from Settings::Ai::ep
///// Datei: src/ai/aop_controller.cpp

#include "pch.hpp"
#include "ai/aop_controller.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "renderer_state.hpp"

namespace Repl { namespace Policy {

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state) {
    (void)fctx; (void)state;
    Decision d{};

    if constexpr (Settings::Ai::enabled && Settings::Ai::aopEnabled) {
        // Einzige Quelle für [REPL/POLICY]-Logs: an Perf-Cadence gekoppelt
        if constexpr (Settings::PerfLog::enabled) {
            const int warm = Settings::PerfLog::warmupFrames;
            const int step = Settings::PerfLog::everyN;
            if (state.frameCount > warm && (state.frameCount % step) == 0) {
                LUCHS_LOG_HOST("[REPL/POLICY] evaluate (stub, ep=%s)", Settings::Ai::ep);
            }
        } else {
            // Falls Perf-Logs aus: sehr sparsam loggen (alle 120 Frames)
            static int s_last = -1000000000;
            if (state.frameCount - s_last >= 120) {
                LUCHS_LOG_HOST("[REPL/POLICY] evaluate (stub, ep=%s)", Settings::Ai::ep);
                s_last = state.frameCount;
            }
        }
    }

    return d;
}

}} // namespace Repl::Policy
