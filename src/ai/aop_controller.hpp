///// OtterDream — Replikatoren
///// File: src/ai/aop_controller.hpp
///// Purpose: AOP-Controller – Feature→Policy (Ordnung/Rebase/Enable) – Stubs
///// Phase: 2 (Policy-Replikatoren)
///// Hooks: frame_pipeline (nach ensureAnalysisMetrics())
///// Depends: Forward-Decls (FrameContext, RendererState)
///// Build: /WX-safe
///// Log-Tags: [REPL/POLICY]
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Decision-Struktur minimal; echte Features/Batching folgen.

#pragma once

struct FrameContext;
struct RendererState;

namespace Repl { namespace Policy {

    struct Decision {
        int  order = 1;          // 1./2. Ordnung für Perturbation
        bool rebase = false;     // Rebase jetzt?
        bool enablePerturb = false;
    };

    // Evaluierung pro Frame/Tile-Gruppe (Stub).
    Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state);

}} // namespace Repl::Policy
