///// OtterDream — Replikatoren
///// File: src/ai/aop_controller.cpp
///// Purpose: AOP-Controller – Feature->Policy (Ordnung/Rebase/Enable) – Stubs
///// Phase: 2 (Policy-Replikatoren)
///// Hooks: frame_pipeline (nach ensureAnalysisMetrics())
///// Depends: pch.hpp, luchs_log_host.hpp, settings.hpp, ai/aop_controller.hpp
///// Build: /WX-safe
///// Log-Tags: [REPL/POLICY]
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Nutzt Settings::Ai.* nur für Logs; echte Features folgen.

#include "pch.hpp"
#include "ai/aop_controller.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

namespace Repl { namespace Policy {

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state) {
    (void)fctx; (void)state;
    Decision d{};
    if constexpr (Settings::Ai::enabled && Settings::Ai::aopEnabled) {
        LUCHS_LOG_HOST("[REPL/POLICY] evaluate policy (stub, ep=%s)", Settings::Ai::ep);
    }
    return d;
}

}} // namespace Repl::Policy
