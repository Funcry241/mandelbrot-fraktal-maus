///// Otter: Orbit-Replikatoren – Gate/Rebase Entry (Stub), /WX-safe.
///// Schneefuchs: Kein Return vor Code (fix für C4702); leichte Includes.
///// Maus: Logtag [REPL/ORBIT]; Hooks in frame_pipeline & renderer_loop.
///// Datei: src/perturb_core.cpp

#include "pch.hpp"
#include "perturb_core.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

struct FrameContext;
struct RendererState;

namespace Repl { namespace Orbit {

void maybe_prepare_orbit(FrameContext& fctx, const RendererState& state) {
    (void)fctx; (void)state;

    // Compile-time gate ohne unreachable-Pfad: nur loggen, wenn Feature aktiv ist.
    if constexpr (Settings::Perturb::enabled) {
        LUCHS_LOG_HOST("[REPL/ORBIT] gatePixelSize=%d deltaScale=%.3f sandboxTile=%d (stub)",
                       Settings::Perturb::gatePixelSize,
                       static_cast<double>(Settings::Perturb::deltaScale),
                       Settings::Perturb::sandboxTile);
    }
}

}} // namespace Repl::Orbit
