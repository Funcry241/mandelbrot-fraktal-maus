///// Otter: Orbit replicators — gate/rebase entry (stub); perf-cadenced logs
///// Schneefuchs: /WX-safe; compile-time gate; no unreachable paths
///// Maus: Tag [REPL/ORBIT]; cadence = Settings::PerfLog (warmup + everyN)
///// Datei: src/perturb_core.cpp

#include "pch.hpp"
#include "perturb_core.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include "renderer_state.hpp"
#include "frame_context.hpp"

namespace Repl { namespace Orbit {

void maybe_prepare_orbit(FrameContext& fctx, const RendererState& state) {
    (void)fctx;

    // Keine Logs, wenn globales PerfLogging aus ist.
    if constexpr (Settings::Perturb::enabled
                  && Settings::performanceLogging && Settings::PerfLog::enabled) {
        const int warm = Settings::PerfLog::warmupFrames;
        const int step = Settings::PerfLog::everyN;
        if (state.frameCount > warm && (state.frameCount % step) == 0) {
            LUCHS_LOG_HOST("[REPL/ORBIT] gatePixelSize=%d deltaScale=%.3f sandboxTile=%d (stub)",
                           Settings::Perturb::gatePixelSize,
                           static_cast<double>(Settings::Perturb::deltaScale),
                           Settings::Perturb::sandboxTile);
        }
    }
}

}} // namespace Repl::Orbit
