///// Otter: AOP controller — single source for [REPL/POLICY] logging cadence
///// Schneefuchs: Canonical 4-line header; /WX-safe; forward decls only
///// Maus: Use Settings::PerfLog cadence; no duplicate logs in frame pipeline
///// Datei: src/ai/aop_controller.hpp

#pragma once

struct FrameContext;
struct RendererState;

namespace Repl { namespace Policy {

    struct Decision {
        int  order = 1;          // 1./2. Ordnung für Perturbation (stub)
        bool rebase = false;     // Rebase jetzt? (stub)
        bool enablePerturb = false;
    };

    // Evaluierung pro Frame (Stub). Enthält die EINZIGE [REPL/POLICY]-Logik.
    Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state);

}} // namespace Repl::Policy
