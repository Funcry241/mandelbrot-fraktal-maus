///// Otter: Zoom V9.1 — PN + Alpha-Beta mit erweiterter Perf-Telemetrie (eine Zeile, selten, ASCII, stabil).
///// Schneefuchs: MSVC /WX clean; keine ungenutzten Symbole; deterministisches dt-Clamp; API unverändert.
///// Maus: „erst zentrieren, dann zoomen“ — Winner-Tile → (nx,ny) filtern → PN-Pan; Telemetrie zeigt alles Wichtige.
///*** Datei: src/zoom_logic.cpp

#pragma warning(push)
#pragma warning(disable: 4100) // API-Parameter evtl. ungenutzt

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
// Strategie:
//   1) Winner-Take-All misst pro Frame (nx, ny) des besten Tiles.
//   2) Alpha-Beta-Filter (CV-Modell) glättet Position & leitet eine sanfte Zielgeschwindigkeit ab.
//   3) Pan via Proportional Navigation (PN) killt Tangentialanteile (Orbit).
//   4) Zoom zweistufig: „Far-Approach“ (kleines Grund-Zoom), „Near-Center“ (Cone-Gate mit Hysterese).
namespace Tuned {
    // Alpha-Beta-Filter (constant-velocity)
    inline constexpr float kAlpha            = 0.35f;  // 0..1  (Positions-Korrektur)
    inline constexpr float kBeta             = 0.12f;  // 0..1  (Geschwindigkeits-Korrektur)

    // Mess-Gate (zu große Sprünge nur gedämpft übernehmen)
    inline constexpr float kJumpSoft         = 0.40f;  // NDC, ab hier Alpha weich herunterskalieren
    inline constexpr float kJumpHard         = 0.90f;  // NDC, oberhalb stark dämpfen (Outlier)

    // PN-Pan (kein Kreisen)
    inline constexpr float kKp               = 1.60f;  // radial:  v_r = -Kp * e
    inline constexpr float kKn               = 1.00f;  // tangential: v_t = -Kn * (dtheta/dt)*r*n_perp
    inline constexpr float kPanVelLerp       = 0.30f;  // EMA auf Pan-Geschwindigkeit
    inline constexpr float kPanVelMax        = 0.50f;  // Deckel [NDC/s]
    inline constexpr float kPanDeadband      = 0.004f; // Ruheband gegen Mikrowobble

    // Zoom (logarithmische Rate, d.h. zoom *= exp(rate*dt))
    // Phase 1 (weit weg): sanftes Grundzoom, um schneller „ins Geschehen“ zu kommen.
    inline constexpr float kZoomFarMin       = 0.15f;  // log-zoom/s, wenn r >= kConeEnter
    // Phase 2 (nahe Zentrum): Cone-Gate mit Hysterese + quadratische Annäherungsformel
    inline constexpr float kConeEnter        = 0.16f;  // Zoomen EIN, wenn |nx|,|ny| <= 0.16
    inline constexpr float kConeExit         = 0.22f;  // Zoomen AUS, wenn > 0.22
    inline constexpr float kZoomNearGain     = 0.55f;  // log-zoom/s bei r→0
    inline constexpr float kZoomVelLerp      = 0.25f;  // EMA auf log-zoom-Rate

    // Sicheres dt
    inline constexpr float kDtMin            = 1.0f/200.0f;
    inline constexpr float kDtMax            = 1.0f/30.0f;

    // Telemetrie
    inline constexpr int   kPerfEveryN       = 16;
}

// =============================== State ======================================
struct State {
    // Gefiltertes Ziel (NDC) und Ableitungen
    bool     filtInit   = false;
    float    tx         = 0.0f;  // gefilterte Zielposition x (NDC)
    float    ty         = 0.0f;  // gefilterte Zielposition y (NDC)
    float    tvx        = 0.0f;  // gefilterte Zielgeschw. x (NDC/s)
    float    tvy        = 0.0f;  // gefilterte Zielgeschw. y (NDC/s)

    // LOS-Winkel für PN
    bool     prevThetaInit = false;
    float    prevTheta     = 0.0f;

    // Ausgabe-Geschwindigkeiten (geglättet)
    float    panVX      = 0.0f;  // [NDC/s]
    float    panVY      = 0.0f;  // [NDC/s]
    float    zoomV      = 0.0f;  // [log-zoom/s]

    // Zoom-Gate Status
    bool     canZoom    = false;

    // Telemetrie
    uint64_t frameIdx   = 0;
    int      prevCand   = -1;
};
static State g;

// ============================== Helpers =====================================
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float absf  (float v) { return v < 0.0f ? -v : v; }
static inline float hypotf(float x, float y) { return std::sqrt(x*x + y*y); }
static inline float rad2deg(float r){ return r * 57.29577951308232f; }

static inline void indexToXY(int idx, int tilesX, int& tx, int& ty) {
    tx = (idx % tilesX);
    ty = (idx / tilesX);
}
static inline void tileCenterNDC(int tx, int ty, int tilesX, int tilesY, float& nx, float& ny) {
    const float fx = (static_cast<float>(tx) + 0.5f) / static_cast<float>(tilesX);
    const float fy = (static_cast<float>(ty) + 0.5f) / static_cast<float>(tilesY);
    nx = 2.0f * fx - 1.0f;
    ny = 2.0f * fy - 1.0f;
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
            LUCHS_LOG_HOST("[ZoomV9] ERROR stats size mismatch: need=%d e=%zu c=%zu statsPx=%d",
                           n, frameCtx.entropy.size(), frameCtx.contrast.size(), statsPx);
        }
        return;
    }

    // --------- Winner-Take-All Messung ----------
    int   bestIdx = -1;
    float bestVal = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        const float v = frameCtx.entropy[static_cast<size_t>(i)] + frameCtx.contrast[static_cast<size_t>(i)];
        if (v > bestVal) { bestVal = v; bestIdx = i; }
    }
    zs.hadCandidate = (bestIdx >= 0 && bestVal > 0.0f);

    float measNX = 0.0f, measNY = 0.0f;
    if (zs.hadCandidate) {
        int tx, ty; indexToXY(bestIdx, tilesX, tx, ty);
        tileCenterNDC(tx, ty, tilesX, tilesY, measNX, measNY);
    }

    // --------- Zeitnormierung ----------
    const float dtRaw = (frameCtx.deltaSeconds > 0.0f ? frameCtx.deltaSeconds : 1.0f/60.0f);
    const float dt    = clampf(dtRaw, Tuned::kDtMin, Tuned::kDtMax);

    // --------- Alpha-Beta-Filter (auf Messung in NDC) ----------
    if (!g.filtInit) {
        if (zs.hadCandidate) {
            g.tx = measNX; g.ty = measNY; g.tvx = 0.0f; g.tvy = 0.0f;
            g.filtInit = true;
            g.prevThetaInit = false;
        }
    } else {
        // Prädiktion
        const float px = g.tx + g.tvx * dt;
        const float py = g.ty + g.tvy * dt;

        float zx = px, zy = py; // Standard: keine Messung → einfach vorhersagen
        if (zs.hadCandidate) {
            // Sprungdämpfung: sehr große Messsprünge nur abgeschwächt einmischen
            const float jump = hypotf(measNX - px, measNY - py);
            float a = Tuned::kAlpha;
            if (jump > Tuned::kJumpSoft) {
                // weich auf 10..30% des ursprünglichen Alpha dämpfen
                const float t = clampf((jump - Tuned::kJumpSoft) / (Tuned::kJumpHard - Tuned::kJumpSoft), 0.0f, 1.0f);
                const float scale = 0.3f - 0.2f * t; // 0.3 → 0.1
                a *= scale;
            }
            const float b = Tuned::kBeta;

            // Korrektur
            const float rx = measNX - px;
            const float ry = measNY - py;
            zx = px + a * rx;
            zy = py + a * ry;
            const float dvx = (b * rx) / dt;
            const float dvy = (b * ry) / dt;

            g.tvx += dvx;
            g.tvy += dvy;
        }
        g.tx = clampf(zx, -1.2f, +1.2f); // etwas größer als Sichtfeld (sanfte Ränder)
        g.ty = clampf(zy, -1.2f, +1.2f);
    }

    // --------- PN-Pan: v = -Kp*e - Kn*dot(theta)*r*n_perp ----------
    float ex = g.tx;
    float ey = g.ty;
    const float r  = hypotf(ex, ey);

    // Debug-/Perf-Variablen für Logging
    float theta = 0.0f;
    float losRate = 0.0f;      // [rad/s]
    float vX = 0.0f, vY = 0.0f;
    float vMagPre = 0.0f;
    int   vCapHit = 0;

    if (g.filtInit && r > 1e-5f) {
        theta = std::atan2(ey, ex);
        float dtheta = 0.0f;
        if (!g.prevThetaInit) {
            g.prevTheta = theta;
            g.prevThetaInit = true;
        } else {
            dtheta = theta - g.prevTheta;
            // wrap to (-pi, pi]
            while (dtheta >  3.14159265f) dtheta -= 6.28318531f;
            while (dtheta <= -3.14159265f) dtheta += 6.28318531f;
            g.prevTheta = theta;
        }
        losRate = dtheta / dt;               // [rad/s]
        const float nx = -ey / r;            // n_perp.x
        const float ny =  ex / r;            // n_perp.y

        const float vPX = -Tuned::kKp * ex;                 // radial
        const float vPY = -Tuned::kKp * ey;
        const float vTX = -Tuned::kKn * losRate * r * nx;   // tangentiale Dämpfung
        const float vTY = -Tuned::kKn * losRate * r * ny;

        vX = vPX + vTX;
        vY = vPY + vTY;

        // Deadband nahe Zentrum
        if (r <= Tuned::kPanDeadband) { vX = 0.0f; vY = 0.0f; }

        // Clamp & EMA
        vMagPre = hypotf(vX, vY);
        if (vMagPre > Tuned::kPanVelMax) { vCapHit = 1; }
        if (vMagPre > Tuned::kPanVelMax) {
            const float s = Tuned::kPanVelMax / (vMagPre + 1e-9f);
            vX *= s; vY *= s;
        }
        g.panVX = (1.0f - Tuned::kPanVelLerp) * g.panVX + Tuned::kPanVelLerp * vX;
        g.panVY = (1.0f - Tuned::kPanVelLerp) * g.panVY + Tuned::kPanVelLerp * vY;
    } else {
        // kein Ziel / nahe Zentrum: auslaufen lassen
        g.panVX *= 0.85f;
        g.panVY *= 0.85f;
    }

    // --------- Zoom zweistufig ----------
    // Phase 1: weit weg → sanftes Grundzoom, um schneller in interessante Strukturen zu kommen
    float zTarget = (!g.filtInit ? 0.0f : Tuned::kZoomFarMin);

    // Phase 2: nah am Zentrum → Cone-Gate + quadratischer Ramp-Up
    const bool insideCone = (absf(g.tx) <= Tuned::kConeEnter && absf(g.ty) <= Tuned::kConeEnter);
    const bool outsideCone= (absf(g.tx) >= Tuned::kConeExit  || absf(g.ty) >= Tuned::kConeExit );
    if (!g.canZoom && insideCone) g.canZoom = true;
    if ( g.canZoom && outsideCone) g.canZoom = false;

    if (g.canZoom) {
        const float gateR   = Tuned::kConeEnter;
        const float rr      = clampf(r / gateR, 0.0f, 1.0f);
        const float weight  = (1.0f - rr);          // 0..1
        zTarget = Tuned::kZoomNearGain * weight * weight; // je näher am Zentrum, desto höher die Rate (bis kZoomNearGain)
    }

    g.zoomV = (1.0f - Tuned::kZoomVelLerp) * g.zoomV + Tuned::kZoomVelLerp * zTarget;

    // --------- Anwenden ----------
    RS_OFFSET_X(rs) += g.panVX * dt;
    RS_OFFSET_Y(rs) += g.panVY * dt;
    RS_ZOOM(rs)     *= std::exp(g.zoomV * dt);

    // --------- Performance-Telemetrie (kompakt, alle kPerfEveryN Frames) ----------
    if constexpr (Settings::performanceLogging) {
        if ((g.frameIdx % Tuned::kPerfEveryN) == 0) {
            const int cand = zs.hadCandidate ? 1 : 0;
            const float dtMs = dt * 1000.0f;
            const float vMag = hypotf(g.panVX, g.panVY);
            const int tilesN = n;
            // Eine Zeile, alles Wichtige:
            // f, dt, statsPx, tiles, idx, bestScore, cand, meas(x,y), filt(x,y), r, theta(deg), los(deg/s),
            // panV(x,y)|mag, capHit, cone, zTarget, zoomV, zoom, tv(x,y)
            LUCHS_LOG_HOST("[ZoomV9TL] f=%llu dt=%.1fms statsPx=%d tiles=%d idx=%d best=%.3f cand=%d "
                           "meas=(%.3f,%.3f) filt=(%.3f,%.3f) r=%.4f th=%.1fdeg los=%.1fdeg/s "
                           "panV=(%.4f,%.4f)|%.4f cap=%d cone=%d zt=%.3f zv=%.3f zoom=%.6f tv=(%.3f,%.3f)",
                           (unsigned long long)g.frameIdx, dtMs, statsPx, tilesN, bestIdx, bestVal, cand,
                           measNX, measNY, g.tx, g.ty, r, rad2deg(theta), rad2deg(losRate),
                           g.panVX, g.panVY, vMag, vCapHit, (g.canZoom?1:0), zTarget, g.zoomV, RS_ZOOM(rs),
                           g.tvx, g.tvy);
        }
        const int candNow = zs.hadCandidate ? 1 : 0;
        if (g.prevCand != candNow) {
            LUCHS_LOG_HOST("[ZoomV9][CAND] cand=%d statsPx=%d", candNow, statsPx);
            g.prevCand = candNow;
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
