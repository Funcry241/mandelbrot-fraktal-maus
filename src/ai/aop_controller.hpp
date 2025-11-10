///// Otter: AOP controller — single source for [REPL/POLICY] logging cadence (shadow only), feature packing
///// Schneefuchs: Canonical 4-line header; /WX-safe; forward decls only; header/source in sync
///// Maus: Uses Settings::PerfLog cadence; no duplicate logs in frame pipeline; statsPx comes from FrameContext
///// Datei: src/ai/aop_controller.hpp

#pragma once

struct FrameContext;
struct RendererState;

namespace Repl { namespace Policy {

    struct Decision {
        int  order = 1;          // 1./2. Ordnung (stub)
        bool rebase = false;     // Rebase jetzt? (stub)
        bool enablePerturb = false;
    };

    // Evaluierung pro Frame (Stub). Enthält die EINZIGE [REPL/POLICY]-Logik (shadow-only).
    Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state);

}} // namespace Repl::Policy
