///// Otter: Zoom v7 — Lock+Cone: winner-take-all target, cone-gated zoom, critically-damped pan.
///** Schneefuchs: MSVC /WX clean; no unused; ASCII logs only; API & headers unverändert.
///** Maus: ruhig, zielstrebig: erst zentrieren, dann zoomen. Keine Kreiserei.

#pragma warning(push)
#pragma warning(disable: 4100)

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
// Steuerung: "erst zentrieren, dann zoomen"
namespace Tuned {
    // Score/Lock
    inline constexpr int   kStickFrames        = 48;     // ~0.8s @60fps min Haltezeit
    inline constexpr float kScoreMargin        = 1.12f;  // neuer Kandidat muss 12% besser sein

    // Center-Cone (NDC)
    inline constexpr float kConeEnter          = 0.12f;  // Zoomen EIN wenn |nx|,|ny| < 0.12
    inline constexpr float kConeExit           = 0.18f;  // Zoomen AUS wenn > 0.18 (Hysterese)

    // Pan-Regler (kritisch gedämpft "light")
    inline constexpr float kPanP               = 2.0f;   // Zielgeschw. [NDC/s] pro 1.0 NDC Fehler
    inline constexpr float kPanVelLerp         = 0.25f;  // EMA auf Geschw.
    inline constexpr float kPanVelMax          = 0.80f;  // Deckel [NDC/s]
    inline constexpr float kPanDeadband        = 0.004f; // Ruheband gegen Mikrowobble

    // Zoom (logarithmische Geschwindigkeit)
    inline constexpr float kZoomInRate         = 0.55f;  // log-zoom / s, wenn im Konus
    inline constexpr float kZoomVelLerp        = 0.25f;  // EMA auf log-zoom-Rate
    inline constexpr float kZoomStopRate       = 0.0f;   // ansonsten 0 (kein Zoomen außerhalb Konus)

    // Sicheres dt
    inline constexpr float kDtMin              = 1.0f/200.0f;
    inline constexpr float kDtMax              = 1.0f/30.0f;

    // Telemetrie
    inline constexpr int   kPerfEveryN         = 16;
}

struct State {
    // Target lock
    int      lockIdx     = -1;
    float    lockScore   = 0.0f;
    int      stickLeft   = 0;

    // Smoothed velocities
    float    panVX       = 0.0f;  // NDC/s
    float    panVY       = 0.0f;  // NDC/s
    float    zoomV       = 0.0f;  // log-zoom / s

    // Status
    bool     canZoom     = false; // im Center-Cone?
    uint64_t frameIdx    = 0;
    int      prevCand    = -1;
};
static State g;

// ============================== Helpers =====================================
static inline float clampf(float v, float a, float b){ return std::max(a, std::min(v, b)); }
static inline float absf(float v){ return v < 0.0f ? -v : v; }

static inline void indexToXY(int idx, int tilesX, int& tx, int& ty){
    tx = (idx % tilesX);
    ty = (idx / tilesX);
}
static inline void tileCenterNDC(int tx, int ty, int tilesX, int tilesY, float& nx, float& ny){
    const float x0 = (tx + 0.5f) / (float)tilesX;
    const float y0 = (ty + 0.5f) / (float)tilesY;
    nx = 2.0f * x0 - 1.0f;
    ny = 2.0f * y0 - 1.0f;
}

// ============================ Public (light) =================================
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
    if(n<=0 || (int)entropy.size()<n || (int)contrast.size()<n){
        state.hadCandidate = false;
        zr.shouldZoom = false;
        zr.bestIndex  = -1;
        zr.newOffsetX = currentOffset.x;
        zr.newOffsetY = currentOffset.y;
        return zr;
    }

    // WTA nur für API-kompatible Rückgabe; echte Anwendung passiert in evaluateAndApply()
    int   bestIdx = -1;
    float bestVal = -std::numeric_limits<float>::infinity();
    for(int i=0;i<n;++i){
        const float v = entropy[(size_t)i] + contrast[(size_t)i];
        if(v > bestVal){ bestVal = v; bestIdx = i; }
    }
    state.hadCandidate = (bestIdx >= 0 && bestVal > 0.0f);
    zr.bestIndex  = (state.hadCandidate? bestIdx : -1);
    zr.shouldZoom = state.hadCandidate;
    zr.newOffsetX = currentOffset.x;
    zr.newOffsetY = currentOffset.y;
    return zr;
}

// =============================== Core =======================================
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

    if ((int)frameCtx.entropy.size() < n || (int)frameCtx.contrast.size() < n) {
        zs.hadCandidate = false;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZoomV7] ERROR stats size mismatch: need=%d e=%zu c=%zu statsPx=%d",
                           n, frameCtx.entropy.size(), frameCtx.contrast.size(), statsPx);
        }
        return;
    }

    // --------- Winner-Take-All + Hysterese Lock ----------
    int   bestIdx = -1;
    float bestVal = -std::numeric_limits<float>::infinity();
    for (int i=0; i<n; ++i){
        const float v = frameCtx.entropy[(size_t)i] + frameCtx.contrast[(size_t)i];
        if (v > bestVal){ bestVal = v; bestIdx = i; }
    }
    zs.hadCandidate = (bestIdx >= 0 && bestVal > 0.0f);

    if (!zs.hadCandidate){
        // Kein Ziel: sanft ausrollen (Pan/Zoom -> 0)
        g.lockIdx   = -1;
        g.lockScore = 0.0f;
        g.stickLeft = 0;
    } else {
        if (g.lockIdx < 0){
            // Kein Lock: neuen nehmen
            g.lockIdx   = bestIdx;
            g.lockScore = bestVal;
            g.stickLeft = Tuned::kStickFrames;
            if constexpr (Settings::performanceLogging){
                LUCHS_LOG_HOST("[ZoomV7] lock idx=%d score=%.3f", g.lockIdx, g.lockScore);
            }
        } else {
            // Lock halten, außer deutlich besserer Kandidat ODER stick abgelaufen
            if (bestIdx != g.lockIdx){
                const bool marginBeat = (bestVal > Tuned::kScoreMargin * g.lockScore);
                if (marginBeat || g.stickLeft <= 0){
                    g.lockIdx   = bestIdx;
                    g.lockScore = bestVal;
                    g.stickLeft = Tuned::kStickFrames;
                    if constexpr (Settings::performanceLogging){
                        LUCHS_LOG_HOST("[ZoomV7] relock idx=%d score=%.3f", g.lockIdx, g.lockScore);
                    }
                } else {
                    // lock halten
                }
            } else {
                // gleicher Kandidat – Score mitziehen
                g.lockScore = bestVal;
            }
        }
    }
    if (g.stickLeft > 0) g.stickLeft--;

    // --------- Zielposition (NDC) des gelockten Tiles ----------
    float tgtNX = 0.0f, tgtNY = 0.0f;
    if (g.lockIdx >= 0){
        int tx, ty; indexToXY(g.lockIdx, tilesX, tx, ty);
        tileCenterNDC(tx, ty, tilesX, tilesY, tgtNX, tgtNY);
    }

    // --------- Regelung: Pan (kritisch gedämpft "light") ----------
    const float errX = tgtNX;
    const float errY = tgtNY;
    float errMag = std::sqrt(errX*errX + errY*errY);

    // Zielgeschwindigkeit proportional zum Fehler (ease out Richtung Zentrum)
    float vTX = 0.0f, vTY = 0.0f;
    if (errMag > Tuned::kPanDeadband){
        vTX = Tuned::kPanP * errX; // [NDC/s]
        vTY = Tuned::kPanP * errY;
        // Clamp
        const float vMag = std::sqrt(vTX*vTX + vTY*vTY);
        if (vMag > Tuned::kPanVelMax){
            const float s = Tuned::kPanVelMax / (vMag + 1e-9f);
            vTX *= s; vTY *= s;
        }
    } else {
        vTX = 0.0f; vTY = 0.0f;
    }

    // EMA auf Geschwindigkeit (entspricht Dämpfung)
    g.panVX = (1.0f - Tuned::kPanVelLerp) * g.panVX + Tuned::kPanVelLerp * vTX;
    g.panVY = (1.0f - Tuned::kPanVelLerp) * g.panVY + Tuned::kPanVelLerp * vTY;

    // --------- Center-Cone-Gate für Zoom ----------
    const bool insideCone = (absf(tgtNX) <= Tuned::kConeEnter && absf(tgtNY) <= Tuned::kConeEnter);
    const bool outsideCone= (absf(tgtNX) >= Tuned::kConeExit  || absf(tgtNY) >= Tuned::kConeExit );

    if (!g.canZoom && insideCone) g.canZoom = true;
    if ( g.canZoom && outsideCone) g.canZoom = false;

    const float zTarget = (g.canZoom ? Tuned::kZoomInRate : Tuned::kZoomStopRate);
    g.zoomV = (1.0f - Tuned::kZoomVelLerp) * g.zoomV + Tuned::kZoomVelLerp * zTarget;

    // --------- Zeitnormierung ----------
    const float dtRaw = (frameCtx.deltaSeconds > 0.0f ? frameCtx.deltaSeconds : 1.0f/60.0f);
    const float dt    = clampf(dtRaw, Tuned::kDtMin, Tuned::kDtMax);

    // --------- Anwenden ----------
    RS_OFFSET_X(rs) += g.panVX * dt;              // NDC
    RS_OFFSET_Y(rs) += g.panVY * dt;
    RS_ZOOM(rs)     *= std::exp(g.zoomV * dt);    // log-zoom

    // --------- Telemetrie ----------
    if constexpr (Settings::performanceLogging) {
        if ((g.frameIdx % Tuned::kPerfEveryN) == 0) {
            const float best = (g.lockIdx>=0 ? g.lockScore : 0.0f);
            LUCHS_LOG_HOST("[ZoomV7] f=%llu statsPx=%d n=%d best=%.4f panV=(%.4f,%.4f) cone=%d zoom=%.6f",
                           (unsigned long long)g.frameIdx, statsPx, n, best,
                           g.panVX, g.panVY, g.canZoom ? 1 : 0, RS_ZOOM(rs));
        }
        const int cand = zs.hadCandidate ? 1 : 0;
        if (g.prevCand != cand) {
            LUCHS_LOG_HOST("[ZoomV7][CAND] cand=%d statsPx=%d", cand, statsPx);
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
