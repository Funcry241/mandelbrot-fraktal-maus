///// Otter: Rullmolder Step 1 — blunt zoom (depth only) + foundational telemetry.
///** Schneefuchs: Minimal invasive; no pan/targets; /WX clean; safe casts (no ref-casts).
///// Maus: Stable ASCII keys; rate-limited; dt-invariant math; pch first.
///// Datei: src/zoom_logic.cpp

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter (API preserved)

#include "pch.hpp"
#include "zoom_logic.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <type_traits> // remove_reference_t, remove_cv_t

// Access helpers consistent with existing state
#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace ZoomLogic {

// --- tiny helpers ------------------------------------------------------------

static inline float get_dt_seconds(const FrameContext& fc) noexcept {
    // Minimal guard: default ~60 FPS if dt invalid.
    return (fc.deltaSeconds > 0.0f) ? fc.deltaSeconds : (1.0f / 60.0f);
}

static inline double blunt_zoom_rate_per_sec() noexcept {
    // Fixed rate for Step 1: +5% per second. No Settings involved.
    return 0.05; // 0.05 == +5%/s
}

// --- local telemetry state ---------------------------------------------------

struct ZLogState {
    uint64_t frame = 0;
    bool     headerPrinted = false;
};
static ZLogState zls;

// --- public surface (API preserved) ------------------------------------------

ZoomResult evaluateTarget(const std::vector<float>& /*entropy*/,
                          const std::vector<float>& /*contrast*/,
                          int /*tilesX*/, int /*tilesY*/,
                          int /*width*/, int /*height*/,
                          float2 currentOffset, float /*zoom*/,
                          float2 /*previousOffset*/,
                          ZoomState& state) noexcept
{
    // Step 1 has no target logic: no pan, no re-center.
    ZoomResult zr{};
    state.hadCandidate = false;
    zr.shouldZoom      = true;           // depth-only zoom regardless of metrics
    zr.bestIndex       = -1;             // no tile chosen
    zr.newOffsetX      = currentOffset.x;
    zr.newOffsetY      = currentOffset.y;
    return zr;
}

// Internal: apply blunt zoom (depth only, dt-invariant), with foundational telemetry.
static void update(FrameContext& frameCtx, RendererState& rs, ZoomState& /*zs*/)
{
    zls.frame++;

    const float  dt   = get_dt_seconds(frameCtx);
    const double rate = blunt_zoom_rate_per_sec();

    // Safely derive scalar type of zoom without cv/ref
    using ZoomT = std::remove_cv_t<std::remove_reference_t<decltype(RS_ZOOM(rs))>>;

    const ZoomT  z0  = static_cast<ZoomT>(RS_ZOOM(rs));                // old zoom (copied)
    const double ldz = rate * static_cast<double>(dt);                 // log delta this frame
    const double g   = std::exp(ldz);                                  // growth multiplier
    const ZoomT  z1  = static_cast<ZoomT>(static_cast<double>(z0) * g);// new zoom (typed)

    // Write back (assignment to the actual lvalue; no ref static_cast)
    RS_ZOOM(rs) = z1;

    // --- Foundational Zoom Telemetry (rate-limited, header once) -------------
    if constexpr (Settings::ZoomLog::enabled) {
        const bool needHeader = (Settings::ZoomLog::header && !zls.headerPrinted);
        const uint64_t modN   = (Settings::ZoomLog::everyN > 0)
                              ? static_cast<uint64_t>(Settings::ZoomLog::everyN)
                              : 1ULL;
        const bool emitData   = ((zls.frame % modN) == 0);

        if (needHeader) {
            LUCHS_LOG_HOST("[ZHDR] keys=f,dt_ms,z0,z1,g,rps,ldz%s",
                           Settings::ZoomLog::includeCenter ? ",cx,cy" : "");
            zls.headerPrinted = true;
        }

        if (emitData) {
            const double dt_ms = static_cast<double>(dt) * 1000.0;
            if (Settings::ZoomLog::includeCenter) {
                const double cx = static_cast<double>(RS_OFFSET_X(rs));
                const double cy = static_cast<double>(RS_OFFSET_Y(rs));
                LUCHS_LOG_HOST("[ZLOG][S1] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f cx=%.9f cy=%.9f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g, rate, ldz, cx, cy);
            } else {
                LUCHS_LOG_HOST("[ZLOG][S1] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g, rate, ldz);
            }
        }
    }
}

void evaluateAndApply(FrameContext& frameCtx, RendererState& rs, ZoomState& zs, float dtOverrideSeconds) noexcept
{
    const float savedDt = frameCtx.deltaSeconds;
    if (dtOverrideSeconds > 0.0f) frameCtx.deltaSeconds = dtOverrideSeconds;

    update(frameCtx, rs, zs);

    frameCtx.deltaSeconds = savedDt;
}

} // namespace ZoomLogic

#pragma warning(pop)
