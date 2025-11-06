///// OtterDream — Replikatoren
///// File: src/perturb_core.cpp
///// Purpose: Orbit-Replikatoren (Perturbation) – Gate/Rebase Entry Points (Stubs)
///// Phase: 1 (Orbit-Replikatoren)
///// Hooks: frame_pipeline (vor Compute) ; Hotkeys in renderer_loop (Ctrl+P/F9)
///// Depends: pch.hpp, luchs_log_host.hpp, settings.hpp, perturb_core.hpp
///// Build: /WX-safe, nur leichte Includes
///// Log-Tags: [REPL/ORBIT]  (Alias: [PERT])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Verwendet Settings::Perturb.*; echte Δ/Drift/Rebase folgt in Phase 1.

#include "pch.hpp"
#include "perturb_core.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

struct FrameContext; struct RendererState;

namespace Repl { namespace Orbit {

void maybe_prepare_orbit(FrameContext& fctx, const RendererState& state) {
    (void)fctx; (void)state;
    if constexpr (!Settings::Perturb::enabled) {
        return;
    }
    LUCHS_LOG_HOST("[REPL/ORBIT] gatePixelSize=%d deltaScale=%.3f sandboxTile=%d (stub)",
                   Settings::Perturb::gatePixelSize,
                   (double)Settings::Perturb::deltaScale,
                   Settings::Perturb::sandboxTile);
}

}} // namespace Repl::Orbit
