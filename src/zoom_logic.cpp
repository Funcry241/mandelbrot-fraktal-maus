///// Otter: Rullmolder Step 2 — blunt zoom + gentle nudge toward Interest (dt-invariant).
///** Schneefuchs: Minimal invasive; caps & deadzone; /WX clean; safe casts (no ref-casts).
///// Maus: Stable ASCII keys; rate-limited; pch first; optional logs [ZPAN1]/[ZPERF]/[ZLEASH].
///// Fink: Zoom-korrekte Pan-Umrechnung (px→world per pixelScale/zoom) gegen Überschwinger.
///// Dachs: Quickfix B+ — Axis-weighted Leash (X stärker), weniger Seitwärtsdrift.
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
#include <type_traits>
#include <algorithm>
#include <chrono>

#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace ZoomLogic {

// --- tiny helpers ------------------------------------------------------------

static inline float get_dt_seconds(const FrameContext& fc) noexcept {
    return (fc.deltaSeconds > 0.0f) ? fc.deltaSeconds : (1.0f / 60.0f);
}

static inline double blunt_zoom_rate_per_sec() noexcept {
    return 0.20; // +20%/s
}

// Gentle nudge tunables
struct NudgeCfg {
    double gainPerSec      = 0.95;
    double deadzoneNdc     = 0.08;
    double maxPxPerFrame   = 12.0;
    double yScale          = 1.0;
    double strengthFloor   = 0.40;
};
static constexpr NudgeCfg kNudge{};

// Axis-weighted Leash: X früh/stark bremsen, Y spät/schwach
struct AxisLeashCfg {
    double xStart = 0.20, xStop = 0.55, xMin = 0.05;
    double yStart = 0.40, yStop = 0.90, yMin = 0.35;
};
static constexpr AxisLeashCfg kLeash{};

// --- local telemetry state ---------------------------------------------------

struct ZLogState {
    uint64_t frame = 0;
    bool     headerPrinted = false;
};
static ZLogState zls;

// --- public surface ----------------------------------------------------------

ZoomResult evaluateTarget(const std::vector<float>& /*entropy*/,
                          const std::vector<float>& /*contrast*/,
                          int /*tilesX*/, int /*tilesY*/,
                          int /*width*/, int /*height*/,
                          float2 currentOffset, float /*zoom*/,
                          float2 /*previousOffset*/,
                          ZoomState& state) noexcept
{
    ZoomResult zr{};
    state.hadCandidate = false;
    zr.shouldZoom      = true;
    zr.bestIndex       = -1;
    zr.newOffsetX      = currentOffset.x;
    zr.newOffsetY      = currentOffset.y;
    return zr;
}

// --- core --------------------------------------------------------------------

static void update(FrameContext& frameCtx, RendererState& rs, ZoomState& /*zs*/)
{
    using Clock = std::chrono::steady_clock;
    [[maybe_unused]] const auto tUpdateStart = Clock::now();
    long long pan_us = 0;

    zls.frame++;

    const float  dt   = get_dt_seconds(frameCtx);
    const double rate = blunt_zoom_rate_per_sec();

    using ZoomT = std::remove_cv_t<std::remove_reference_t<decltype(RS_ZOOM(rs))>>;
    const ZoomT  z0  = static_cast<ZoomT>(RS_ZOOM(rs));
    const double ldz = rate * static_cast<double>(dt);
    const double g   = std::exp(ldz);
    const ZoomT  z1  = static_cast<ZoomT>(static_cast<double>(z0) * g);
    RS_ZOOM(rs) = z1;

    // Logging cadence (einmal definieren, überall nutzen)
    const uint64_t modN       = (Settings::ZoomLog::everyN > 0)
                              ? static_cast<uint64_t>(Settings::ZoomLog::everyN) : 1ULL;
    const bool     emitEveryN = ((zls.frame % modN) == 0);

    // -------------------- Gentle Nudge (PAN) ---------------------------------
    if (rs.interest.valid && rs.width > 0 && rs.height > 0) {
        [[maybe_unused]] const auto tPanStart = Clock::now();

        const double ndcX_raw = rs.interest.ndcX;
        const double ndcY_raw = rs.interest.ndcY;

        auto applyDeadzone = [](double v, double dz)->double {
            const double a = std::abs(v);
            if (a <= dz) return 0.0;
            const double t = std::min(1.0, (a - dz) / (1.0 - dz));
            return (v < 0.0) ? -t : t;
        };
        double ndcX = applyDeadzone(ndcX_raw, kNudge.deadzoneNdc);
        double ndcY = applyDeadzone(ndcY_raw, kNudge.deadzoneNdc);

        // ---- Axis-weighted radial leash (B+) ----
        auto smooth01 = [](double x, double a, double b)->double{
            if (x <= a) return 0.0;
            if (x >= b) return 1.0;
            const double t = (x - a) / (b - a);
            return t*t*(3.0 - 2.0*t);
        };
        auto leashAxis = [&](double a, double s, double e, double minF)->double{
            const double r = std::abs(a);
            return std::max(minF, 1.0 - smooth01(r, s, e));
        };
        const double leashX = leashAxis(ndcX, kLeash.xStart, kLeash.xStop, kLeash.xMin);
        const double leashY = leashAxis(ndcY, kLeash.yStart, kLeash.yStop, kLeash.yMin);
        ndcX *= leashX;
        ndcY *= leashY;

        if constexpr (Settings::ZoomLog::enabled) {
            if (emitEveryN && (leashX < 0.999 || leashY < 0.999)) {
                LUCHS_LOG_HOST("[ZLEASH] f=%llu leashX=%.2f leashY=%.2f ndc'=(%.3f,%.3f)",
                               (unsigned long long)zls.frame, leashX, leashY, ndcX, ndcY);
            }
        }
        // -----------------------------------------

        const bool hitDZ_X = (std::abs(ndcX_raw) <= kNudge.deadzoneNdc);
        const bool hitDZ_Y = (std::abs(ndcY_raw) <= kNudge.deadzoneNdc);

        if (ndcX != 0.0 || ndcY != 0.0) {
            const double s = std::max(kNudge.strengthFloor, std::min(1.0, rs.interest.strength));

            const double dx_px_goal = ndcX * 0.5 * static_cast<double>(rs.width);
            const double dy_px_goal = ndcY * 0.5 * static_cast<double>(rs.height);

            const double alpha = 1.0 - std::exp(-(kNudge.gainPerSec * s) * static_cast<double>(dt));

            const double step_px_x_raw = dx_px_goal * alpha;
            const double step_px_y_raw = dy_px_goal * alpha * kNudge.yScale;

            auto cap = [](double v, double cap)->double {
                if (v >  cap) return cap;
                if (v < -cap) return -cap;
                return v;
            };
            double step_px_x = cap(step_px_x_raw, kNudge.maxPxPerFrame);
            double step_px_y = cap(step_px_y_raw, kNudge.maxPxPerFrame);

            const bool hitCAP_X = (step_px_x != step_px_x_raw);
            const bool hitCAP_Y = (step_px_y != step_px_y_raw);

            const double psx = static_cast<double>(rs.pixelScale.x);
            const double psy = static_cast<double>(rs.pixelScale.y);
            const bool scaleZero = (psx == 0.0 && psy == 0.0);

            if (!scaleZero) {
                const double invZ = (RS_ZOOM(rs) != 0.0) ? (1.0 / static_cast<double>(RS_ZOOM(rs))) : 0.0;
                const double dWorldX = step_px_x * psx * invZ;
                const double dWorldY = step_px_y * psy * invZ;

                RS_OFFSET_X(rs) += dWorldX;
                RS_OFFSET_Y(rs) += dWorldY;

                const int flags = (hitDZ_X ? 1 : 0)
                                | (hitDZ_Y ? 2 : 0)
                                | (hitCAP_X ? 4 : 0)
                                | (hitCAP_Y ? 8 : 0)
                                | (scaleZero ? 16 : 0);

                if constexpr (Settings::ZoomLog::enabled) {
                    if (emitEveryN) {
                        LUCHS_LOG_HOST("[ZPAN1] f=%llu ndc=(%.4f,%.4f) a=%.3f s=%.2f "
                                       "goal_px=(%.2f,%.2f) step_px=(%.2f,%.2f) dWorld=(%.9f,%.9f) invZ=%.6g flags=0x%02X",
                                       (unsigned long long)zls.frame,
                                       ndcX_raw, ndcY_raw, alpha, s,
                                       dx_px_goal, dy_px_goal, step_px_x, step_px_y,
                                       dWorldX, dWorldY, invZ, flags);
                    }
                }
            }

            pan_us += (long long)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tPanStart).count();
        }
    }

    // --- Foundational Zoom Telemetry -----------------------------------------
    if constexpr (Settings::ZoomLog::enabled) {
        const bool needHeader = (Settings::ZoomLog::header && !zls.headerPrinted);
        if (needHeader) {
            LUCHS_LOG_HOST("[ZHDR] keys=f,dt_ms,z0,z1,g,rps,ldz%s",
                           Settings::ZoomLog::includeCenter ? ",cx,cy" : "");
            zls.headerPrinted = true;
        }

        if (emitEveryN) {
            const double dt_ms = static_cast<double>(dt) * 1000.0;
            if (Settings::ZoomLog::includeCenter) {
                const double cx = static_cast<double>(RS_OFFSET_X(rs));
                const double cy = static_cast<double>(RS_OFFSET_Y(rs));
                LUCHS_LOG_HOST("[ZLOG][S2] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f cx=%.9f cy=%.9f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g, rate, ldz, cx, cy);
            } else {
                LUCHS_LOG_HOST("[ZLOG][S2] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g, rate, ldz);
            }

            if (rs.interest.valid) {
                LUCHS_LOG_HOST("[ZPAN0] f=%llu interest ndc=(%.6f,%.6f) R=%.4f s=%.2f",
                               (unsigned long long)zls.frame,
                               rs.interest.ndcX, rs.interest.ndcY,
                               rs.interest.radiusNdc, rs.interest.strength);
            }

            const long long update_us =
                (long long)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tUpdateStart).count();
            LUCHS_LOG_HOST("[ZPERF] f=%llu update_us=%lld pan_us=%lld",
                           (unsigned long long)zls.frame, update_us, pan_us);
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
