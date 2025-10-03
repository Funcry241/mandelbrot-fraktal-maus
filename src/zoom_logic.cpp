///// Otter: Rullmolder Step 2 — blunt zoom + gentle nudge toward Interest (dt-invariant).
///** Schneefuchs: Minimal invasive; caps & deadzone; /WX clean; safe casts (no ref-casts).
///// Maus: Stable ASCII keys; rate-limited; pch first; optional logs [ZPAN1].
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
#include <algorithm>

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
    // Fixed rate for "depth only" part: +5% per second.
    return 0.05; // 0.05 == +5%/s
}

// Gentle nudge tunables (keine Settings-Abhängigkeit für schnellen Test)
struct NudgeCfg {
    double gainPerSec      = 1.25; // wie schnell wir den Offset "einholen" (~72% in 1 s)
    double deadzoneNdc     = 0.08; // um 0..±0.08 NDC keine Bewegung (Jitterfrei)
    double maxPxPerFrame   = 12.0; // harte Kappe pro Frame (unabhängig von gain) – sanft!
    double yScale          = 1.0;  // evtl. leicht <1.0 wenn y nervös ist; hier neutral
    double strengthFloor   = 0.40; // minimale Gewichtung, falls Interest.strength sehr klein ist
};
static constexpr NudgeCfg kNudge{};

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
    // Kein Target-Pick in diesem Step; nur Tiefen-Zoom + sanfter Nudge im update().
    ZoomResult zr{};
    state.hadCandidate = false;
    zr.shouldZoom      = true;
    zr.bestIndex       = -1;
    zr.newOffsetX      = currentOffset.x;
    zr.newOffsetY      = currentOffset.y;
    return zr;
}

// Internal: apply blunt zoom (depth only) and gentle nudge toward Interest.
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

    // --- Logging cadence for this frame (used by [ZPAN1] and [ZLOG]) ---------
    const uint64_t modN      = (Settings::ZoomLog::everyN > 0)
                             ? static_cast<uint64_t>(Settings::ZoomLog::everyN)
                             : 1ULL;
    const bool     emitEveryN = ((zls.frame % modN) == 0);

    // --- Gentle Nudge (PAN) --------------------------------------------------
    // Wir verschieben die Center-Koordinaten minimal Richtung Interest (Sticker),
    // indem wir den Bildschirm-Versatz (in px) -> Welt (PixelScale) abbilden.
    if (rs.interest.valid && rs.width > 0 && rs.height > 0) {
        // Entfernung des Stickers vom Bildschirmzentrum in NDC [-1..1]
        double ndcX = rs.interest.ndcX;
        double ndcY = rs.interest.ndcY;

        // Deadzone: kleine Abweichungen ignorieren (Jitter vermeiden)
        auto applyDeadzone = [](double v, double dz)->double {
            const double a = std::abs(v);
            if (a <= dz) return 0.0;
            // sanfter Übergang: linear ab (a - dz) / (1 - dz) mit ursprünglichem Vorzeichen
            const double t = std::min(1.0, (a - dz) / (1.0 - dz));
            return (v < 0.0) ? -t : t;
        };
        ndcX = applyDeadzone(ndcX, kNudge.deadzoneNdc);
        ndcY = applyDeadzone(ndcY, kNudge.deadzoneNdc);

        if (ndcX != 0.0 || ndcY != 0.0) {
            // Interesse-Gewichtung: Floor, damit bei schwachen Werten nicht "tot"
            const double s = std::max(kNudge.strengthFloor, std::min(1.0, rs.interest.strength));

            // Ziel-Offset in Pixeln (vom Center aus): NDC -> px
            const double dx_px_goal = ndcX * 0.5 * static_cast<double>(rs.width);
            const double dy_px_goal = ndcY * 0.5 * static_cast<double>(rs.height);

            // Exponentielle Annäherung: alpha = 1 - exp(-gain*dt)
            const double alpha = 1.0 - std::exp(-(kNudge.gainPerSec * s) * static_cast<double>(dt));

            // Roh-Schritt in Pixeln
            double step_px_x = dx_px_goal * alpha;
            double step_px_y = dy_px_goal * alpha * kNudge.yScale;

            // Harte Kappe pro Frame (sanft!) – erst in Pixeln begrenzen, dann in Welt umrechnen
            auto cap = [](double v, double cap)->double {
                if (v >  cap) return cap;
                if (v < -cap) return -cap;
                return v;
            };
            step_px_x = cap(step_px_x, kNudge.maxPxPerFrame);
            step_px_y = cap(step_px_y, kNudge.maxPxPerFrame);

            // Pixel -> Welt (Komplexebene) per PixelScale
            const double2 ps = rs.pixelScale; // Schrittweite pro Pixel in Weltkoordinaten
            if (ps.x != 0.0 || ps.y != 0.0) {
                const double dWorldX = step_px_x * static_cast<double>(ps.x);
                const double dWorldY = step_px_y * static_cast<double>(ps.y);

                // Center in Richtung Sticker ziehen
                RS_OFFSET_X(rs) += dWorldX;
                RS_OFFSET_Y(rs) += dWorldY;

                if constexpr (Settings::ZoomLog::enabled) {
                    if (emitEveryN) {
                        LUCHS_LOG_HOST("[ZPAN1] ndc=(%.4f,%.4f) a=%.3f s=%.2f step_px=(%.2f,%.2f) dWorld=(%.9f,%.9f)",
                                       rs.interest.ndcX, rs.interest.ndcY, alpha, s,
                                       step_px_x, step_px_y, dWorldX, dWorldY);
                    }
                }
            }
        }
    }

    // --- Foundational Zoom Telemetry (rate-limited, header once) -------------
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

            // Optional Interest-Log (zusätzlich zu [ZPAN1])
            if (rs.interest.valid) {
                LUCHS_LOG_HOST("[ZPAN0] interest ndc=(%.6f,%.6f) R=%.4f s=%.2f",
                               rs.interest.ndcX, rs.interest.ndcY,
                               rs.interest.radiusNdc, rs.interest.strength);
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
