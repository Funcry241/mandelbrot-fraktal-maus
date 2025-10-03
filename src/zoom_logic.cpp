///// Otter: Zoom V8 — Proportional Navigation (PN): kein Kreisen, erst zentrieren, dann zoomen.
///*** Schneefuchs: MSVC /WX clean; ASCII-Logs; API unverändert; deterministisches dt-Clamp.
///*** Maus: ruhig & zielstrebig: Winner-Lock + Cone-Hysterese + PN-Dämpfung der Blickwinkel-Rate.
///*** Datei: src/zoom_logic.cpp

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter (API beibehalten)

#include "zoom_logic.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace ZoomLogic {

// ============================== Tunables ====================================
// Strategie: Winner-Take-All + Lock, Pan via Proportional Navigation (PN),
// Zoomen gated über Center-Cone mit Hysterese.
namespace Tuned {
    // Target-Lock (Winner-Take-All + Hysterese)
    inline constexpr int   kStickFrames        = 48;     // ~0.8s @60fps Mindesthaltezeit
    inline constexpr float kScoreMargin        = 1.12f;  // neuer Kandidat muss 12% besser sein

    // Center-Cone (NDC)
    inline constexpr float kConeEnter          = 0.12f;  // Zoomen EIN wenn |nx|,|ny| < 0.12
    inline constexpr float kConeExit           = 0.18f;  // Zoomen AUS wenn > 0.18

    // PN-Pan-Regler
    inline constexpr float kKp                 = 2.00f;  // Positionsfehler → Wunschgeschwindigkeit [NDC/s]
    inline constexpr float kKn                 = 1.00f;  // Dämpfung auf LOS-Winkelrate (tangentiale Komponente)
    inline constexpr float kPanVelLerp         = 0.25f;  // EMA auf Pan-Geschwindigkeit
    inline constexpr float kPanVelMax          = 0.80f;  // Deckel [NDC/s]
    inline constexpr float kPanDeadband        = 0.004f; // Ruheband gegen Mikrowobble

    // Zoom (logarithmische Geschwindigkeit)
    inline constexpr float kZoomGain           = 0.55f;  // log-zoom / s bei r→0
    inline constexpr float kZoomVelLerp        = 0.25f;  // EMA auf log-zoom-Rate

    // Sicheres dt
    inline constexpr float kDtMin              = 1.0f/200.0f;
    inline constexpr float kDtMax              = 1.0f/30.0f;

    // Telemetrie
    inline constexpr int   kPerfEveryN         = 16;
}

// =============================== State ======================================
struct State {
    // Target lock
    int      lockIdx     = -1;
    float    lockScore   = 0.0f;
    int      stickLeft   = 0;

    // Pan / Zoom velocities (geglättet)
    float    panVX       = 0.0f;  // [NDC/s]
    float    panVY       = 0.0f;  // [NDC/s]
    float    zoomV       = 0.0f;  // [log-zoom/s]

    // LOS-Winkel (für PN)
    float    prevTheta   = 0.0f;
    bool     prevThetaInit = false;

    // Zoom-Gate Status
    bool     canZoom     = false;

    // Telemetrie
    uint64_t frameIdx    = 0;
    int      prevCand    = -1;
};
static State g;

// ============================== Helpers =====================================
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float absf  (float v) { return v < 0.0f ? -v : v; }

static inline void indexToXY(int idx, int tilesX, int& tx, int& ty) {
    tx = (idx % tilesX);
    ty = (idx / tilesX);
}
static inline void tileCenterNDC(int tx, int ty, int tilesX, int tilesY, float& nx, float& ny) {
    const float x0 = (static_cast<float>(tx) + 0.5f) / static_cast<float>(tilesX);
    const float y0 = (static_cast<float>(ty) + 0.5f) / static_cast<float>(tilesY);
    nx = 2.0f * x0 - 1.0f;
    ny = 2.0f * y0 - 1.0f;
}

// ===========================================================================
// Öffentliche, leichte Zielermittlung (nur für API; Anwendung in evaluateAndApply)
// ===========================================================================
ZoomResult evaluateTarget(const std::vector<float>& entropy,
                          const std::vector<float>& contrast,
                          int tilesX, int tilesY,
                          int /*width*/, int /*height*/,
                          float2 currentOffset, float /*zoom*/,
                          float2 /*previousOffset*/,
                          ZoomState& state) noexcept
{
    ZoomResult zr{};
    const int n = tilesX * tilesY;
    if (n <= 0 || static_cast<int>(entropy.size()) < n || static_cast<int>(contrast.size()) < n) {
        state.hadCandidate = false;
        zr.shouldZoom = false;
        zr.bestIndex  = -1;
        zr.newOffsetX = currentOffset.x;
        zr.newOffsetY = currentOffset.y;
        return zr;
    }

    int   bestIdx = -1;
    float bestVal = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        const float v = entropy[static_cast<size_t>(i)] + contrast[static_cast<size_t>(i)];
        if (v > bestVal) { bestVal = v; bestIdx = i; }
    }

    state.hadCandidate = (bestIdx >= 0 && bestVal > 0.0f);
    zr.bestIndex  = (state.hadCandidate ? bestIdx : -1);
    zr.shouldZoom = state.hadCandidate;
    zr.newOffsetX = currentOffset.x;
    zr.newOffsetY = currentOffset.y;
    return zr;
}

// ================================ Core ======================================
static void update(FrameContext& frameCtx, RendererState& rs, ZoomState& zs)
{
    g.frameIdx++;

    const int width  = frameCtx.width;
    const int height = frameCtx.height;
    if (width <= 0 || height <= 0) { zs.hadCandidate = false; return; }

    const int statsPx = (frameCtx.statsTileSize > 0 ? frameCtx.statsTileSize : frameCtx.tileSize);
    if (statsPx <= 0) { zs.hadCandidate = false; return; }

    const int tilesX = (width  + statsPx - 1) / statsPx;
    const int tilesY = (height + statsPx - 1) / statsPx;
    const int n      = tilesX * tilesY;

    if (static_cast<int>(frameCtx.entropy.size()) < n || static_cast<int>(frameCtx.contrast.size()) < n) {
        zs.hadCandidate = false;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZoomV8] ERROR stats size mismatch: need=%d e=%zu c=%zu statsPx=%d",
                           n, frameCtx.entropy.size(), frameCtx.contrast.size(), statsPx);
        }
        return;
    }

    // --------- Winner-Take-All + Hysterese Lock ----------
    int   bestIdx = -1;
    float bestVal = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        const float v = frameCtx.entropy[static_cast<size_t>(i)] + frameCtx.contrast[static_cast<size_t>(i)];
        if (v > bestVal) { bestVal = v; bestIdx = i; }
    }
    zs.hadCandidate = (bestIdx >= 0 && bestVal > 0.0f);

    if (!zs.hadCandidate) {
        // Kein Ziel: Lock lösen, sanft ausrollen (Pan/Zoom → 0)
        g.lockIdx     = -1;
        g.lockScore   = 0.0f;
        g.stickLeft   = 0;
    } else {
        if (g.lockIdx < 0) {
            g.lockIdx   = bestIdx;
            g.lockScore = bestVal;
            g.stickLeft = Tuned::kStickFrames;
            g.prevThetaInit = false; // LOS neu initialisieren
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[ZoomV8] lock idx=%d score=%.3f", g.lockIdx, g.lockScore);
            }
        } else {
            if (bestIdx != g.lockIdx) {
                const bool marginBeat = (bestVal > Tuned::kScoreMargin * g.lockScore);
                if (marginBeat || g.stickLeft <= 0) {
                    g.lockIdx   = bestIdx;
                    g.lockScore = bestVal;
                    g.stickLeft = Tuned::kStickFrames;
                    g.prevThetaInit = false;
                    if constexpr (Settings::performanceLogging) {
                        LUCHS_LOG_HOST("[ZoomV8] relock idx=%d score=%.3f", g.lockIdx, g.lockScore);
                    }
                } // sonst Lock halten
            } else {
                g.lockScore = bestVal; // Score mitziehen
            }
        }
    }
    if (g.stickLeft > 0) g.stickLeft--;

    // --------- Zielposition (NDC) des gelockten Tiles ----------
    float tgtNX = 0.0f, tgtNY = 0.0f;
    if (g.lockIdx >= 0) {
        int tx, ty; indexToXY(g.lockIdx, tilesX, tx, ty);
        tileCenterNDC(tx, ty, tilesX, tilesY, tgtNX, tgtNY);
    }

    // --------- Zeitnormierung ----------
    const float dtRaw = (frameCtx.deltaSeconds > 0.0f ? frameCtx.deltaSeconds : 1.0f/60.0f);
    const float dt    = clampf(dtRaw, Tuned::kDtMin, Tuned::kDtMax);

    // --------- PN-Pan: v = -Kp*e - Kn*dot(theta)*r*n_perp ----------
    const float ex = tgtNX, ey = tgtNY;
    const float r  = std::sqrt(ex*ex + ey*ey);
    float vX = 0.0f, vY = 0.0f;

    if (r > 1e-5f) {
        const float theta = std::atan2(ey, ex);
        float dtheta = 0.0f;
        if (!g.prevThetaInit) {
            g.prevTheta = theta;
            g.prevThetaInit = true;
            dtheta = 0.0f;
        } else {
            dtheta = theta - g.prevTheta;
            // wrap to (-pi, pi]
            while (dtheta >  3.14159265f) dtheta -= 6.28318531f;
            while (dtheta <= -3.14159265f) dtheta += 6.28318531f;
            g.prevTheta = theta;
        }
        const float losRate = dtheta / dt;     // [rad/s]
        const float nx = -ey / r;              // n_perp.x
        const float ny =  ex / r;              // n_perp.y

        const float vPX = -Tuned::kKp * ex;                // radial (Proportional)
        const float vPY = -Tuned::kKp * ey;
        const float vTX = -Tuned::kKn * losRate * r * nx;  // tangentiale Dämpfung
        const float vTY = -Tuned::kKn * losRate * r * ny;

        vX = vPX + vTX;
        vY = vPY + vTY;

        // Deadband nahe Zentrum
        if (r <= Tuned::kPanDeadband) { vX = 0.0f; vY = 0.0f; }

        // Clamp & EMA
        const float vMag = std::sqrt(vX*vX + vY*vY);
        if (vMag > Tuned::kPanVelMax) {
            const float s = Tuned::kPanVelMax / (vMag + 1e-9f);
            vX *= s; vY *= s;
        }

        g.panVX = (1.0f - Tuned::kPanVelLerp) * g.panVX + Tuned::kPanVelLerp * vX;
        g.panVY = (1.0f - Tuned::kPanVelLerp) * g.panVY + Tuned::kPanVelLerp * vY;
    } else {
        // nahe Zentrum: auslaufen lassen
        g.panVX *= 0.85f;
        g.panVY *= 0.85f;
    }

    // --------- Center-Cone-Gate für Zoom (Hysterese) ----------
    const bool insideCone = (absf(tgtNX) <= Tuned::kConeEnter && absf(tgtNY) <= Tuned::kConeEnter);
    const bool outsideCone= (absf(tgtNX) >= Tuned::kConeExit  || absf(tgtNY) >= Tuned::kConeExit );
    if (!g.canZoom && insideCone) g.canZoom = true;
    if ( g.canZoom && outsideCone) g.canZoom = false;

    float zTarget = 0.0f;
    if (g.canZoom) {
        const float gateR   = Tuned::kConeEnter;
        const float rr      = clampf(r / gateR, 0.0f, 1.0f);
        const float weight  = (1.0f - rr);          // 0..1
        zTarget = Tuned::kZoomGain * weight * weight;
    } else {
        zTarget = 0.0f;
    }
    g.zoomV = (1.0f - Tuned::kZoomVelLerp) * g.zoomV + Tuned::kZoomVelLerp * zTarget;

    // --------- Anwenden ----------
    RS_OFFSET_X(rs) += g.panVX * dt;
    RS_OFFSET_Y(rs) += g.panVY * dt;
    RS_ZOOM(rs)     *= std::exp(g.zoomV * dt);

    // --------- Telemetrie ----------
    if constexpr (Settings::performanceLogging) {
        if ((g.frameIdx % Tuned::kPerfEveryN) == 0) {
            const float best = (g.lockIdx >= 0 ? g.lockScore : 0.0f);
            LUCHS_LOG_HOST("[ZoomV8] f=%llu statsPx=%d n=%d lock=%d best=%.4f r=%.4f panV=(%.4f,%.4f) cone=%d zoom=%.6f",
                           (unsigned long long)g.frameIdx, statsPx, n, g.lockIdx, best, r,
                           g.panVX, g.panVY, g.canZoom ? 1 : 0, RS_ZOOM(rs));
        }
        const int cand = zs.hadCandidate ? 1 : 0;
        if (g.prevCand != cand) {
            LUCHS_LOG_HOST("[ZoomV8][CAND] cand=%d statsPx=%d", cand, statsPx);
            g.prevCand = cand;
        }
    }
}

// ---------------------------------------------------------------------------
void evaluateAndApply(FrameContext& frameCtx, RendererState& rs, ZoomState& zs, float dtOverrideSeconds) noexcept
{
    const float savedDt = frameCtx.deltaSeconds;
    if (dtOverrideSeconds > 0.0f) frameCtx.deltaSeconds = dtOverrideSeconds;

    update(frameCtx, rs, zs);

    frameCtx.deltaSeconds = savedDt;
}

} // namespace ZoomLogic
#pragma warning(pop)
