///// Otter: Zoom V9.2 — Alpha-Beta + Proportional Navigation + Far-Zoom-Ramp + Winkeltreue Relock.
///** Schneefuchs: MSVC /WX clean; kompakte Perf-Telemetrie; deterministisches dt-Clamp; API unverändert.
///** Maus: „erst zentrieren, dann zoomen“ – stuck-frei durch Far-Zoom, kreisfrei durch PN, ruhig durch AB-Filter.
///** Datei: src/zoom_logic.cpp

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

namespace Tuned {
    // Alpha-Beta (constant-velocity) – glättet Messung und leitet Zielgeschw. ab
    inline constexpr float kAlpha            = 0.35f;
    inline constexpr float kBeta             = 0.12f;

    // Messsprünge weich abfangen
    inline constexpr float kJumpSoft         = 0.40f;
    inline constexpr float kJumpHard         = 0.90f;

    // Winner-Lock Hysterese
    inline constexpr int   kStickFrames      = 72;      // ↑ gegenüber V7/V8: weniger Relocks
    inline constexpr float kScoreMargin      = 1.20f;   // ↑: neuer Kandidat muss 20% besser
    inline constexpr float kRelockAngleDeg   = 30.0f;   // nur relocken, wenn Richtung halbwegs ähnlich

    // PN-Pan
    inline constexpr float kKp               = 1.80f;
    inline constexpr float kKn               = 1.00f;
    inline constexpr float kPanVelLerp       = 0.30f;
    inline constexpr float kPanVelMax        = 0.65f;   // ↑ vs. V9.1 (0.50)
    inline constexpr float kPanDeadband      = 0.004f;

    // Zoom
    inline constexpr float kConeEnter        = 0.16f;   // Near-Zoom Gate EIN
    inline constexpr float kConeExit         = 0.22f;   // Near-Zoom Gate AUS
    inline constexpr float kZoomNearGain     = 0.55f;   // im Cone (quadratisch in (1 - r/R))
    inline constexpr float kZoomFarGain      = 0.25f;   // NEU: sanftes Fern-Zoom proportional zu r
    inline constexpr float kZoomVelLerp      = 0.25f;

    // dt-Clamp
    inline constexpr float kDtMin            = 1.0f/200.0f;
    inline constexpr float kDtMax            = 1.0f/30.0f;

    // Perf-Log
    inline constexpr int   kPerfEveryN       = 16;
}

struct State {
    // Filterzustand
    bool     filtInit       = false;
    float    tx             = 0.0f, ty = 0.0f;     // gefiltertes Ziel (NDC)
    float    tvx            = 0.0f, tvy = 0.0f;    // gefilterte Zielgeschwindigkeit

    // Winner-Lock
    int      lockIdx        = -1;
    float    lockScore      = 0.0f;
    int      stickLeft      = 0;

    // Richtungs-/PN
    bool     prevThetaInit  = false;
    float    prevTheta      = 0.0f;

    // Ausgabe
    float    panVX          = 0.0f, panVY = 0.0f;  // [NDC/s]
    float    zoomV          = 0.0f;                // [log-zoom/s]
    bool     canZoom        = false;

    // Telemetrie
    uint64_t frameIdx       = 0;
    int      prevCand       = -1;
    bool     helloDone      = false;
};
static State g;

// helpers
static inline float clampf(float v, float a, float b){ return v<a? a : (v>b? b : v); }
static inline float absf  (float v){ return v<0.0f? -v : v; }
static inline float hypotf(float x,float y){ return std::sqrt(x*x+y*y); }
[[maybe_unused]] static inline float rad2deg(float r){ return r * 57.2957795f; }
[[maybe_unused]] static inline float deg2rad(float d){ return d * 0.01745329252f; }

static inline void indexToXY(int idx, int tilesX, int& tx, int& ty){ tx=idx%tilesX; ty=idx/tilesX; }
static inline void tileCenterNDC(int tx,int ty,int tilesX,int tilesY,float& nx,float& ny){
    const float fx = (tx + 0.5f) / (float)tilesX;
    const float fy = (ty + 0.5f) / (float)tilesY;
    nx = 2.0f*fx - 1.0f; ny = 2.0f*fy - 1.0f;
}

// --- API: leichtgewichtige Zielermittlung (nur Rückgabe, Anwendung in update) ---
ZoomResult evaluateTarget(const std::vector<float>& entropy,
                          const std::vector<float>& contrast,
                          int tilesX, int tilesY,
                          int /*width*/, int /*height*/,
                          float2 currentOffset, float /*zoom*/,
                          float2 /*previousOffset*/,
                          ZoomState& state) noexcept
{
    ZoomResult zr{};
    const int n = tilesX*tilesY;
    if(n<=0 || (int)entropy.size()<n || (int)contrast.size()<n){
        state.hadCandidate=false; zr.shouldZoom=false; zr.bestIndex=-1;
        zr.newOffsetX=currentOffset.x; zr.newOffsetY=currentOffset.y; return zr;
    }
    int bi=-1; float bv=-std::numeric_limits<float>::infinity();
    for(int i=0;i<n;++i){ const float v=entropy[(size_t)i]+contrast[(size_t)i]; if(v>bv){bv=v; bi=i;} }
    state.hadCandidate=(bi>=0 && bv>0.0f);
    zr.bestIndex=(state.hadCandidate? bi : -1); zr.shouldZoom=state.hadCandidate;
    zr.newOffsetX=currentOffset.x; zr.newOffsetY=currentOffset.y; return zr;
}

// --- Kern ---
static void update(FrameContext& frameCtx, RendererState& rs, ZoomState& zs)
{
    g.frameIdx++;

    // einmalige Versionsmeldung im Perf-Mode (damit du sofort siehst, dass V9.2 läuft)
    if constexpr (Settings::performanceLogging) {
        if (!g.helloDone) {
            LUCHS_LOG_HOST("[ZoomV9HELLO] v=9.2 PN+AB+FarZoom kPanMax=%.2f cone=(%.2f/%.2f)",
                           Tuned::kPanVelMax, Tuned::kConeEnter, Tuned::kConeExit);
            g.helloDone = true;
        }
    }

    const int width=frameCtx.width, height=frameCtx.height;
    if(width<=0 || height<=0){ zs.hadCandidate=false; return; }
    const int statsPx = (frameCtx.statsTileSize>0? frameCtx.statsTileSize: frameCtx.tileSize);
    if(statsPx<=0){ zs.hadCandidate=false; return; }

    const int tilesX=(width +statsPx-1)/statsPx;
    const int tilesY=(height+statsPx-1)/statsPx;
    const int n=tilesX*tilesY;

    if((int)frameCtx.entropy.size()<n || (int)frameCtx.contrast.size()<n){
        zs.hadCandidate=false;
        if constexpr(Settings::debugLogging){
            LUCHS_LOG_HOST("[ZoomV9] ERROR stats size mismatch: need=%d e=%zu c=%zu statsPx=%d",
                           n, frameCtx.entropy.size(), frameCtx.contrast.size(), statsPx);
        }
        return;
    }

    // Winner-Take-All
    int   bestIdx=-1; float bestVal=-std::numeric_limits<float>::infinity();
    for(int i=0;i<n;++i){ const float v=frameCtx.entropy[(size_t)i]+frameCtx.contrast[(size_t)i];
        if(v>bestVal){ bestVal=v; bestIdx=i; } }
    zs.hadCandidate=(bestIdx>=0 && bestVal>0.0f);

    float measNX=0.0f, measNY=0.0f;
    if(zs.hadCandidate){ int tx,ty; indexToXY(bestIdx,tilesX,tx,ty); tileCenterNDC(tx,ty,tilesX,tilesY,measNX,measNY); }

    // dt (einmal berechnen, später wiederverwenden)
    const float dtRaw = (frameCtx.deltaSeconds>0.0f? frameCtx.deltaSeconds : 1.0f/60.0f);
    const float dt    = clampf(dtRaw, Tuned::kDtMin, Tuned::kDtMax);

    // Lock mit Score- UND Winkelhysterese
    auto angleOf = [](float x, float y){ return std::atan2(y,x); };
    if (!zs.hadCandidate){
        g.lockIdx=-1; g.lockScore=0.0f; g.stickLeft=0;
    } else {
        if (g.lockIdx<0){
            g.lockIdx=bestIdx; g.lockScore=bestVal; g.stickLeft=Tuned::kStickFrames; g.prevThetaInit=false;
        } else if (bestIdx!=g.lockIdx){
            bool allow=false;
            if (bestVal > Tuned::kScoreMargin * g.lockScore) {
                // Richtungsprüfung: neuer Vektor vs. aktueller gefilterter
                float th_new = angleOf(measNX,measNY);
                float th_cur = angleOf(g.tx,g.ty);
                float d = th_new - th_cur;
                while(d> 3.14159265f) d-=6.28318531f;
                while(d<-3.14159265f) d+=6.28318531f;
                allow = (std::fabs(d) <= deg2rad(Tuned::kRelockAngleDeg));
            }
            if (allow || g.stickLeft<=0){
                g.lockIdx=bestIdx; g.lockScore=bestVal; g.stickLeft=Tuned::kStickFrames; g.prevThetaInit=false;
            }
        } else {
            g.lockScore=bestVal;
        }
    }
    if (g.stickLeft>0) g.stickLeft--;

    // Alpha-Beta Filter
    if(!g.filtInit){
        if(zs.hadCandidate){ g.tx=measNX; g.ty=measNY; g.tvx=0.0f; g.tvy=0.0f; g.filtInit=true; g.prevThetaInit=false; }
    } else {
        const float px=g.tx + g.tvx*dt, py=g.ty + g.tvy*dt;
        float zx=px, zy=py;
        if(zs.hadCandidate){
            const float jump=hypotf(measNX-px, measNY-py);
            float a=Tuned::kAlpha;
            if(jump > Tuned::kJumpSoft){
                const float t=clampf((jump-Tuned::kJumpSoft)/(Tuned::kJumpHard-Tuned::kJumpSoft),0.0f,1.0f);
                a *= (0.3f - 0.2f*t); // 0.3→0.1
            }
            const float b=Tuned::kBeta;
            const float rx=measNX-px, ry=measNY-py;
            zx = px + a*rx; zy = py + a*ry;
            g.tvx += (b*rx)/dt; g.tvy += (b*ry)/dt;
        }
        g.tx=clampf(zx,-1.2f,+1.2f); g.ty=clampf(zy,-1.2f,+1.2f);
    }

    // PN-Pan
    float ex=g.tx, ey=g.ty; const float r=hypotf(ex,ey);
    float theta=0.0f, losRate=0.0f;
    float vX=0.0f, vY=0.0f, vMagPre=0.0f; int vCapHit=0;

    if(g.filtInit && r>1e-5f){
        theta = std::atan2(ey,ex);
        float dtheta=0.0f;
        if(!g.prevThetaInit){ g.prevTheta=theta; g.prevThetaInit=true; }
        else {
            dtheta = theta - g.prevTheta;
            while(dtheta> 3.14159265f) dtheta-=6.28318531f;
            while(dtheta<=-3.14159265f) dtheta+=6.28318531f;
            g.prevTheta=theta;
        }
        losRate = dtheta / dt;
        const float nx = -ey / r, ny =  ex / r; // n_perp

        const float vPX = -Tuned::kKp * ex;
        const float vPY = -Tuned::kKp * ey;
        const float vTX = -Tuned::kKn * losRate * r * nx;
        const float vTY = -Tuned::kKn * losRate * r * ny;

        vX = vPX + vTX; vY = vPY + vTY;

        if(r <= Tuned::kPanDeadband){ vX=0.0f; vY=0.0f; }

        vMagPre = hypotf(vX,vY); if(vMagPre > Tuned::kPanVelMax) vCapHit=1;
        if(vMagPre > Tuned::kPanVelMax){ const float s=Tuned::kPanVelMax/(vMagPre+1e-9f); vX*=s; vY*=s; }

        g.panVX = (1.0f - Tuned::kPanVelLerp)*g.panVX + Tuned::kPanVelLerp*vX;
        g.panVY = (1.0f - Tuned::kPanVelLerp)*g.panVY + Tuned::kPanVelLerp*vY;

        // Wenn Perf-Logging aus, Warnungen zu ungenutzten Debug-Variablen vermeiden
        if constexpr (!Settings::performanceLogging) { (void)theta; (void)losRate; (void)vCapHit; (void)vMagPre; }
    } else {
        g.panVX*=0.85f; g.panVY*=0.85f;
        if constexpr (!Settings::performanceLogging) { (void)theta; (void)losRate; (void)vCapHit; (void)vMagPre; }
    }

    // Zoom: Far-Ramp + Near-Cone
    const bool insideCone = (absf(g.tx)<=Tuned::kConeEnter && absf(g.ty)<=Tuned::kConeEnter);
    const bool outsideCone= (absf(g.tx)>=Tuned::kConeExit  || absf(g.ty)>=Tuned::kConeExit );
    if(!g.canZoom && insideCone) g.canZoom=true;
    if( g.canZoom && outsideCone) g.canZoom=false;

    float zTarget = 0.0f;
    // Far-Zoom (außerhalb Cone): quadratisch mit r, weicher Start an ConeEnter
    if (!g.canZoom) {
        const float a = clampf((r - Tuned::kConeEnter) / (1.0f - Tuned::kConeEnter), 0.0f, 1.0f);
        zTarget += Tuned::kZoomFarGain * (a*a);
    }
    // Near-Zoom (im Cone): desto näher am Zentrum, desto höher
    if (g.canZoom) {
        const float gateR = Tuned::kConeEnter;
        const float rr    = clampf(r / gateR, 0.0f, 1.0f);
        const float w     = (1.0f - rr);
        zTarget += Tuned::kZoomNearGain * (w*w);
    }

    g.zoomV = (1.0f - Tuned::kZoomVelLerp)*g.zoomV + Tuned::kZoomVelLerp*zTarget;

    // Anwenden (dt wurde oben bereits berechnet)
    RS_OFFSET_X(rs) += g.panVX * dt;
    RS_OFFSET_Y(rs) += g.panVY * dt;
    RS_ZOOM(rs)     *= std::exp(g.zoomV * dt);

    // Performance-Telemetrie (kompakt)
    if constexpr (Settings::performanceLogging) {
        if ((g.frameIdx % Tuned::kPerfEveryN) == 0) {
            const int cand = zs.hadCandidate ? 1 : 0;
            LUCHS_LOG_HOST("[ZoomV9TL] f=%llu dt=%.1fms statsPx=%d tiles=%d idx=%d best=%.3f cand=%d "
                           "meas=(%.3f,%.3f) filt=(%.3f,%.3f) r=%.4f th=%.1fdeg los=%.1fdeg/s "
                           "panV=(%.4f,%.4f)|%.4f cap=%d cone=%d zt=%.3f zv=%.3f zoom=%.6f tv=(%.3f,%.3f)",
                           (unsigned long long)g.frameIdx, dt*1000.0f, statsPx, n, bestIdx, bestVal, cand,
                           measNX, measNY, g.tx, g.ty, r, rad2deg(theta), rad2deg(losRate),
                           g.panVX, g.panVY, hypotf(g.panVX,g.panVY), vCapHit, (g.canZoom?1:0),
                           zTarget, g.zoomV, RS_ZOOM(rs), g.tvx, g.tvy);
        }
        const int candNow = zs.hadCandidate ? 1 : 0;
        if (g.prevCand != candNow) {
            LUCHS_LOG_HOST("[ZoomV9][CAND] cand=%d statsPx=%d", candNow, statsPx);
            g.prevCand = candNow;
        }
    } else {
        // Falls Perf-Logging aus: ungenutzte Helper sauber „referenzieren“, damit MSVC nicht warnt
        (void)&rad2deg; (void)&deg2rad;
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
