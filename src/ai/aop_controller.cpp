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

    // Keine Logs, wenn globales PerfLogging aus ist.
    if constexpr (Settings::Ai::enabled && Settings::Ai::aopEnabled
                  && Settings::performanceLogging && Settings::PerfLog::enabled) {
        const int warm = Settings::PerfLog::warmupFrames;
        const int step = Settings::PerfLog::everyN;
        if (state.frameCount > warm && (state.frameCount % step) == 0) {
            LUCHS_LOG_HOST("[REPL/POLICY] evaluate (stub, ep=%s)", Settings::Ai::ep);
        }
    }

    return d;
}

}} // namespace Repl::Policy
