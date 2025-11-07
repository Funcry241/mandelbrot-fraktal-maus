///// OtterDream — Replikatoren
///// File: src/ai/aop_controller.cpp
///// Purpose: AOP-Controller – Feature->Policy (Ordnung/Rebase/Enable) – Stubs
///// Phase: 2 (Policy-Replikatoren)
///// Hooks: frame_pipeline (nach ensureAnalysisMetrics())
///// Depends: pch.hpp, renderer_state.hpp, luchs_log_host.hpp, settings.hpp, ai/aop_controller.hpp
///// Build: /WX-safe
///// Log-Tags: [REPL/POLICY]
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Nutzt Settings::Ai.* nur für Logs; echte Features folgen.

#include "pch.hpp"
#include "ai/aop_controller.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

namespace Repl { namespace Policy {

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state) {
    (void)fctx; (void)state;
    Decision d{};

    // Throttle noisy boot-stub log: once every 60 frames.
    if constexpr (Settings::Ai::enabled && Settings::Ai::aopEnabled) {
        static int s_lastLogFrame = -1000000000;
        if (state.frameCount - s_lastLogFrame >= 60) {
            LUCHS_LOG_HOST("[REPL/POLICY] evaluate policy (stub, ep=%s)", Settings::Ai::ep);
            s_lastLogFrame = state.frameCount;
        }
    }
    return d;
}

}} // namespace Repl::Policy
