///// OtterDream — Replikatoren
///// File: src/perturb_core.hpp
///// Purpose: Orbit-Replikatoren (Perturbation) - Gate/Rebase Entry Points (Stubs)
///// Phase: 1 (Orbit-Replikatoren)
///// Hooks: frame_pipeline (vor Compute) ; Hotkeys in renderer_loop (Ctrl+P/F9)
///// Depends: FrameContext, RendererState, settings.hpp (im .cpp zusätzlich luchs_log_host.hpp)
///// Build: /WX-safe, keine externen Abhängigkeiten
///// Log-Tags: [REPL/ORBIT]  (Alias: [PERT])
///// Created: 2025-11-06  (Europe/Berlin)
///// Notes: Öffentliche, minimale API. Implementierung in src/perturb_core.cpp.

#pragma once

// Forward Declarations - wir binden hier keine schweren Header ein,
// damit dieses Header leichtgewichtig bleibt und /WX-sicher ist.
struct FrameContext;
struct RendererState;

namespace Repl {
namespace Orbit {

    // Prüft/Gatet vor dem Compute-Launch, bereitet ggf. Orbit-Δ / Rebase vor.
    // Phase-1-Stub: nur Log; echte 1./2.-Ordnung + Drift/Rebase folgt.
    void maybe_prepare_orbit(FrameContext& fctx, const RendererState& state);

} // namespace Orbit
} // namespace Repl
