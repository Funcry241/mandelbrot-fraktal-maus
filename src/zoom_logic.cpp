///// Otter: Rullmolder Step 2 — blunt zoom + gentle nudge toward Interest (dt-invariant).
///** Schneefuchs: Zoom-Hold bei Leash/großem Fehler/schwachem Signal; Logs [ZHOLD]; Caps & Deadzone wie gehabt; /WX clean.
///** Maus: Axis-weighted Leash (X stärker); zoom-korrekte Pan-Umrechnung; stabile ASCII-Logs [ZPAN0/1][ZPERF][ZLEASH].
///// Fink: Zoom-korrekte Pan-Umrechnung (px→world per pixelScale/zoom) gegen Überschwinger.
///// Dachs: Minimal invasiv — nur Reihenfolge geändert (erst Pan/Leash, dann Zoom-Entscheid).
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
    return 0.20; // +20%/s (Basisslope)
}

// Gentle nudge tunables (dt-invariant über alpha)
struct NudgeCfg {
    double gainPerSec      = 1.8;   // schnelleres Nachführen (zeitbasiert)
    double deadzoneNdc     = 0.06;  // früher pannen
    double maxPxPerFrame   = 24.0;  // Panning schafft weite Wege
    double yScale          = 1.00;  // keine Y-Dämpfung
    double strengthFloor   = 0.08;  // kein erzwungener Zoom bei schwachem Signal
};
static constexpr NudgeCfg kNudge{};

// Axis-weighted Leash: X früh/stark bremsen, Y später/schwächer
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

    const float  dt        = get_dt_seconds(frameCtx);
    const double rateBase  = blunt_zoom_rate_per_sec();      // Basisslope (pro Sekunde)
    const double ldz_base  = rateBase * static_cast<double>(dt);
    const double g_base    = std::exp(ldz_base);             // exp(rate*dt)

    // Logging cadence (einmal definieren, überall nutzen)
    const uint64_t modN       = (Settings::ZoomLog::everyN > 0)
                              ? static_cast<uint64_t>(Settings::ZoomLog::everyN) : 1ULL;
    const bool     emitEveryN = ((zls.frame % modN) == 0);

    // --- Wir pannen zuerst (inkl. Leash/Deadzone/Caps) und entscheiden DANACH über den Zoom ---
    // Telemetrie für die spätere Zoom-Entscheidung
    bool   leashActive = false;         // true, wenn Leash oder Caps/DZ greifen
    double errPx       = 1e12;          // Fehler in Pixeln
    bool   weakSignal  = true;          // schwaches Heatmap-Signal
    bool   haveInterest= false;

    // -------------------- Gentle Nudge (PAN) ---------------------------------
    if (rs.interest.valid && rs.width > 0 && rs.height > 0) {
        haveInterest = true;
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

        const bool hitDZ_X = (std::abs(ndcX_raw) <= kNudge.deadzoneNdc);
        const bool hitDZ_Y = (std::abs(ndcY_raw) <= kNudge.deadzoneNdc);

        // Ziel in Pixeln (vor Caps) — dient u. a. zur Zoom-Hold-Entscheidung
        const double dx_px_goal = ndcX * 0.5 * static_cast<double>(rs.width);
        const double dy_px_goal = ndcY * 0.5 * static_cast<double>(rs.height);
        errPx = std::sqrt(dx_px_goal*dx_px_goal + dy_px_goal*dy_px_goal);

        // Stärke des Interesses (gegen erzwungenen Zoom bei schwachem Signal)
        const double s_raw = static_cast<double>(rs.interest.strength);
        const double s     = std::max(kNudge.strengthFloor, std::min(1.0, s_raw));
        weakSignal         = (s_raw < 0.20); // harte Schwelle fürs Zoom-Gating

        // Schrittweite (dt-invariant)
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

        // „Leash aktiv“ wenn Deadzone, Leash-Faktor < 1 oder Caps greifen
        leashActive = (hitDZ_X || hitDZ_Y) || (leashX < 0.999 || leashY < 0.999) || (hitCAP_X || hitCAP_Y);

        // Pan in Weltkoordinaten (pixelScale / zoom)
        const double psx = static_cast<double>(rs.pixelScale.x);
        const double psy = static_cast<double>(rs.pixelScale.y);
        const bool   scaleZero = (psx == 0.0 && psy == 0.0);

        if (!scaleZero) {
            const double invZ = (RS_ZOOM(rs) != 0.0) ? (1.0 / static_cast<double>(RS_ZOOM(rs))) : 0.0;
            const double dWorldX = step_px_x * psx * invZ;
            const double dWorldY = step_px_y * psy * invZ;

            RS_OFFSET_X(rs) += dWorldX;
            RS_OFFSET_Y(rs) += dWorldY;

            if constexpr (Settings::ZoomLog::enabled) {
                if (emitEveryN && (leashX < 0.999 || leashY < 0.999)) {
                    LUCHS_LOG_HOST("[ZLEASH] f=%llu leashX=%.2f leashY=%.2f ndc'=(%.3f,%.3f)",
                                   (unsigned long long)zls.frame, leashX, leashY, ndcX, ndcY);
                }
                if (emitEveryN) {
                    LUCHS_LOG_HOST("[ZPAN1] f=%llu ndc=(%.4f,%.4f) a=%.3f s=%.2f "
                                   "goal_px=(%.2f,%.2f) step_px=(%.2f,%.2f) dWorld=(%.9f,%.9f) invZ=%.6g flags=0x%02X",
                                   (unsigned long long)zls.frame,
                                   ndcX_raw, ndcY_raw, alpha, s,
                                   dx_px_goal, dy_px_goal, step_px_x, step_px_y,
                                   dWorldX, dWorldY, invZ,
                                   (hitDZ_X?1:0)|(hitDZ_Y?2:0)|(hitCAP_X?4:0)|(hitCAP_Y?8:0)|(scaleZero?16:0));
                }
            }
        }

        pan_us += (long long)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tPanStart).count();
    } // end PAN

    // -------------------- Zoom-Entscheidung (Hold vs. Base) -------------------
    using ZoomT = std::remove_cv_t<std::remove_reference_t<decltype(RS_ZOOM(rs))>>;
    const ZoomT z0 = static_cast<ZoomT>(RS_ZOOM(rs));

    // Bedingungen für „Zoom halten“:
    //  - Leash/Caps/DZ greifen (Pan kommt (noch) nicht hinterher)
    //  - Fehler groß (hier: > 25% der kürzeren Bildkante)
    //  - schwaches Signal oder kein Interest
    const double errThreshPx = 0.25 * static_cast<double>(std::min(std::max(rs.width, 0), std::max(rs.height, 0)));
    const bool   holdZoom    = leashActive || (!haveInterest) || weakSignal || (errPx > errThreshPx);

    const double g_applied = holdZoom ? 1.0 : g_base;
    const double ldz_appl  = std::log(g_applied);
    const ZoomT  z1        = static_cast<ZoomT>(static_cast<double>(z0) * g_applied);
    RS_ZOOM(rs) = z1;

    if constexpr (Settings::ZoomLog::enabled) {
        if (emitEveryN && holdZoom) {
            LUCHS_LOG_HOST("[ZHOLD] f=%llu reason=%s%s%s errPx=%.1f thr=%.1f",
                           (unsigned long long)zls.frame,
                           leashActive ? "LEASH " : "",
                           (!haveInterest || weakSignal) ? "WEAK " : "",
                           (errPx > errThreshPx) ? "ERR " : "",
                           errPx, errThreshPx);
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
                               g_applied, rateBase, ldz_appl, cx, cy);
            } else {
                LUCHS_LOG_HOST("[ZLOG][S2] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g_applied, rateBase, ldz_appl);
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
