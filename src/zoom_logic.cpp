///// Otter: Silk-Lite auto-pan/zoom controller (lokal, kantengetrieben, ohne Heatmap-Zwang).
///// Schneefuchs: Sanfte Richtungswechsel via Lock+Cooldown+Hysterese + leichtem Blend; keine API-Drifts.
///@@@ Maus: Ziel = nächstgelegene valide Kachel; Richtung = lokaler Kontrastgradient; ASCII-only Logs.
///// Datei: src/zoom_logic.cpp

#include "zoom_logic.hpp"
#include "frame_context.hpp"    // FrameContext
#include "renderer_state.hpp"   // RendererState
#include "cuda_interop.hpp"     // CudaInterop::getPauseZoom()
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "capybara_mapping.cuh" // capy_pixel_steps_from_zoom_scale(...)

#include <algorithm>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

namespace {

// ----------------------------- Tunables ---------------------------------
// Basis-Schritt (Anteil der Halbbreite/-höhe) etwas kleiner, Kappe unverändert
constexpr float kSEED_STEP_NDC = 0.012f;
constexpr float kSTEP_MAX_NDC  = 0.35f;

// Blend auf Weltzentrum vorsichtiger (weniger Nachschwingen)
constexpr float kBLEND_A       = 0.16f;

// lokale Suche (±R Kacheln)
constexpr int   kSEARCH_R      = 2;

// Kontrast-Schwelle straffer: thr = median + a * MAD
constexpr float kTHR_MED_ADD_MAD = 0.90f;

// Zielstabilisierung (klein & effektiv)
constexpr int   kCOOLDOWN_FRAMES     = 24;    // straffer: min. Frames zwischen Retargets
constexpr float kHYST_FACTOR         = 1.60f; // neuer Score muss 1.6× besser sein
constexpr float kMUCH_BETTER_FACTOR  = 1.90f; // darf Cooldown sofort brechen
constexpr float kMIN_RETARGET_NDC    = 0.10f; // Mindestabstand Lock→Kandidat (NDC)
constexpr float kLOCKBOX_IGNORE_NDC  = 0.18f; // nahe Kandidaten um Lock ignorieren

// Turn-Limiter deutlich straffer
constexpr float kMAX_TURN_DEG        = 15.0f; // max Richtungsänderung/Frame
constexpr float kFLIP_HALVE_DEG      = 90.0f; // starker Flip: Velocity stark dämpfen

// Zusätzliche Bewegungskappe in Bildschirmpixeln (pro Frame)
constexpr double kMAX_PX_MOVE_PER_FRAME = 6.0;

// ------------------------------------------------------------------------

inline float clampf(float x, float a, float b) { return (x < a) ? a : (x > b ? b : x); }
inline bool normalize2D(float& x, float& y) {
    const float n2 = x*x + y*y; if(!(n2>0.f)) return false; const float inv=1.0f/std::sqrt(n2); x*=inv; y*=inv; return true;
}

// Hauptmengen-Test (Cardioid + Periode-2-Bulb)
inline bool insideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25, q = xm*xm + y*y;
    if (q*(q + xm) < 0.25*y*y) return true;
    const double dx = x + 1.0; return (dx*dx + y*y) < 0.0625;
}

// Median/MAD (robust)
float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t m = v.size()/2;
    std::nth_element(v.begin(), v.begin()+m, v.end());
    float med = v[m];
    if ((v.size()%2)==0) {
        std::nth_element(v.begin(), v.begin()+m-1, v.begin()+m);
        med = 0.5f*(med + v[m-1]);
    }
    return med;
}
float mad_from_center_inplace(std::vector<float>& v, float center) {
    for (auto& x : v) x = std::fabs(x - center);
    float mad = median_inplace(v);
    return (mad > 1e-6f) ? mad : 1.0f;
}

// Tileindex → NDC-Zentrum (−1..+1)
inline void tileIndexToNdcCenter(int tilesX, int tilesY, int idx, float& ndcX, float& ndcY) {
    const int tx = (tilesX > 0) ? (idx % tilesX) : 0;
    const int ty = (tilesX > 0) ? (idx / tilesX) : 0;
    const float cx = (static_cast<float>(tx) + 0.5f) / std::max(1, tilesX);
    const float cy = (static_cast<float>(ty) + 0.5f) / std::max(1, tilesY);
    ndcX = cx * 2.0f - 1.0f; ndcY = cy * 2.0f - 1.0f;
}
inline int idxAt(int x, int y, int tilesX) { return y * tilesX + x; }

// edgeScore: lokale Kanten-Plausibilität (Kontrast + Nachbar-Sprung)
inline float edgeScoreAt(const std::vector<float>& contrast, int x, int y, int tilesX, int tilesY){
    const int i = idxAt(x,y,tilesX); const float lc = contrast[(size_t)i];
    float sum=0.f; int n=0;
    const int nx4[4]={x-1,x+1,x,  x}; const int ny4[4]={y,  y,  y-1,y+1};
    for(int k=0;k<4;++k){ const int xn=nx4[k], yn=ny4[k];
        if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        sum += std::fabs(contrast[(size_t)idxAt(xn,yn,tilesX)] - lc); ++n;
    }
    const float neighborDiff = (n>0)?(sum/(float)n):0.f;
    return lc + 0.75f*neighborDiff;
}

// ------------------ Minimaler Ziel-Smoother-Zustand ---------------------
struct ZLock {
    bool  init = false;
    int   lockIndex = -1;
    float lockScore = 0.f;
    float lockNdcX = 0.f, lockNdcY = 0.f;
    int   cooldown = 0;
    float vx = 0.f, vy = 0.f; // NDC-"Velocity" (nur Richtungsmerkung)
};
static ZLock g_zlock;

inline void zs_beginFrame() {
    if (g_zlock.cooldown > 0) g_zlock.cooldown--;
    g_zlock.vx *= 0.9f; g_zlock.vy *= 0.9f; // leichte Dämpfung zur Winkelstabilität
}
inline float angleDeg(float ax, float ay, float bx, float by) {
    const float La2=ax*ax+ay*ay, Lb2=bx*bx+by*by; if(!(La2>0.f)||!(Lb2>0.f)) return 0.f;
    const float La=std::sqrt(La2), Lb=std::sqrt(Lb2);
    float c=(ax*bx+ay*by)/(La*Lb); if(c>1.f)c=1.f; else if(c<-1.f)c=-1.f;
    return std::acos(c)*180.0f/3.14159265358979323846f;
}
inline void lerpDirClamp(float px,float py,float dx,float dy,float t,float& ox,float& oy){
    // einfache, kurze Richtungs-Blend-Variante
    normalize2D(px,py); normalize2D(dx,dy);
    ox = (1.f-t)*px + t*dx; oy = (1.f-t)*py + t*dy; normalize2D(ox,oy);
}

} // namespace

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>& /*entropy*/,
                              const std::vector<float>& contrast,
                              int tilesX, int tilesY,
                              int /*width*/, int /*height*/,
                              float2 /*currentOffset*/, float zoom,
                              float2 previousOffset,
                              ZoomState& /*state*/) noexcept
{
    ZoomResult out{};

    const int total = (tilesX > 0 && tilesY > 0) ? (tilesX * tilesY) : 0;
    if (total <= 0 || static_cast<int>(contrast.size()) < total) {
        out.shouldZoom = Settings::ForceAlwaysZoom;
        return out;
    }

    // Robust-Statistik Kontrast
    std::vector<float> c = contrast;
    const float cMed = median_inplace(c);
    const float cMAD = mad_from_center_inplace(c, cMed);
    const float thr  = cMed + kTHR_MED_ADD_MAD * cMAD;

    // Suche im lokalen Fenster um Bildmitte
    const int cx = tilesX/2, cy = tilesY/2;
    int   bestI=-1; float bestS=-1e30f, bestD2=1e30f; float bestNdcX=0.f, bestNdcY=0.f;

    for (int dy=-kSEARCH_R; dy<=kSEARCH_R; ++dy){
        for (int dx=-kSEARCH_R; dx<=kSEARCH_R; ++dx){
            const int x=cx+dx, y=cy+dy; if(x<0||y<0||x>=tilesX||y>=tilesY) continue;
            const float s = edgeScoreAt(contrast,x,y,tilesX,tilesY); if(!(s>thr)) continue;
            float ndcX=0.f, ndcY=0.f; tileIndexToNdcCenter(tilesX,tilesY,idxAt(x,y,tilesX),ndcX,ndcY);
            const float d2 = ndcX*ndcX + ndcY*ndcY;
            if (d2 < bestD2 - 1e-6f || (std::fabs(d2-bestD2) <= 1e-6f && s > bestS)) {
                bestD2=d2; bestS=s; bestI=idxAt(x,y,tilesX); bestNdcX=ndcX; bestNdcY=ndcY;
            }
        }
    }

    // Erstmaliger Lock
    if (!g_zlock.init && bestI >= 0){
        g_zlock.init = true;
        g_zlock.lockIndex = bestI;
        g_zlock.lockScore = bestS;
        g_zlock.lockNdcX  = bestNdcX;
        g_zlock.lockNdcY  = bestNdcY;
        g_zlock.cooldown  = kCOOLDOWN_FRAMES;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][LOCK][INIT] i=%d ndc=(%.3f,%.3f) score=%.3f thr=%.3f",
                           bestI,(double)bestNdcX,(double)bestNdcY,(double)bestS,(double)thr);
        }
    }

    // Kein Kandidat → weicher Drift
    if (bestI < 0){
        out.shouldZoom = Settings::ForceAlwaysZoom;
        float dirx=1.f, diry=0.f;
        if (g_zlock.init){ dirx=g_zlock.lockNdcX; diry=g_zlock.lockNdcY; normalize2D(dirx,diry); }
        const float invZ=1.0f/std::max(1e-6f,zoom);
        const float step=clampf(kSEED_STEP_NDC,0.f,kSTEP_MAX_NDC);
        out.newOffsetX = previousOffset.x + dirx*(step*invZ);
        out.newOffsetY = previousOffset.y + diry*(step*invZ);
        out.distance   = step*invZ;
        return out;
    }

    // ---- Lock + Cooldown + Hysterese (minimal & wirksam) ----
    const float dLockX = bestNdcX - g_zlock.lockNdcX;
    const float dLockY = bestNdcY - g_zlock.lockNdcY;
    const float distLock = std::sqrt(dLockX*dLockX + dLockY*dLockY);

    const bool farEnough  = distLock >= kMIN_RETARGET_NDC;
    const bool better     = bestS >= g_zlock.lockScore * kHYST_FACTOR;
    const bool muchBetter = bestS >= g_zlock.lockScore * kMUCH_BETTER_FACTOR;
    const bool inLockBox  = distLock <= kLOCKBOX_IGNORE_NDC;

    if (g_zlock.init){
        if ((g_zlock.cooldown <= 0 && farEnough && better) || muchBetter){
            if (!inLockBox || muchBetter){
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ZOOM][RETARGET] old=%d new=%d d=%.3f score=%.3f->%.3f",
                                   g_zlock.lockIndex,bestI,(double)distLock,(double)g_zlock.lockScore,(double)bestS);
                }
                g_zlock.lockIndex = bestI;
                // Score nicht hart setzen → kleine Trägheit gegen Jitter
                g_zlock.lockScore = 0.7f * g_zlock.lockScore + 0.3f * bestS;
                g_zlock.lockNdcX  = bestNdcX;
                g_zlock.lockNdcY  = bestNdcY;
                g_zlock.cooldown  = kCOOLDOWN_FRAMES;
            }
        }
    }

    // Richtung = Gradient am Lock-Index
    const int bx = g_zlock.lockIndex % tilesX, by = g_zlock.lockIndex / tilesX;
    auto ndcOf = [&](int X,int Y){ float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(X,Y,tilesX),nx,ny); return make_float2(nx,ny); };
    const float2 pC = ndcOf(bx,by);

    float2 grad = make_float2(0,0);
    const int nx4b[4]={bx-1,bx+1,bx,  bx};
    const int ny4b[4]={by,  by,  by-1,by+1};
    for (int k=0;k<4;++k){
        const int xn=nx4b[k], yn=ny4b[k];
        if (xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        const float w = contrast[(size_t)idxAt(xn,yn,tilesX)] - contrast[(size_t)g_zlock.lockIndex];
        const float2 pN = ndcOf(xn,yn);
        float vx = pN.x - pC.x, vy = pN.y - pC.y;
        if (normalize2D(vx,vy)) { grad.x += w*vx; grad.y += w*vy; }
    }

    float dirx, diry;
    if (std::fabs(grad.x)+std::fabs(grad.y) > 0.f) { dirx=grad.x; diry=grad.y; normalize2D(dirx,diry); }
    else { dirx=g_zlock.lockNdcX; diry=g_zlock.lockNdcY; if(!normalize2D(dirx,diry)){dirx=1.f; diry=0.f;} }

    // Turn-Limiter: begrenze Winkelwechsel ggü. letzter Richtung
    if (std::fabs(g_zlock.vx)+std::fabs(g_zlock.vy) > 0.f){
        const float turn = angleDeg(g_zlock.vx,g_zlock.vy,dirx,diry);
        if (turn > kMAX_TURN_DEG){
            float ox,oy; const float t = kMAX_TURN_DEG / turn;
            lerpDirClamp(g_zlock.vx,g_zlock.vy,dirx,diry,t,ox,oy);
            dirx=ox; diry=oy;
        }
        if (turn > kFLIP_HALVE_DEG){ g_zlock.vx *= 0.3f; g_zlock.vy *= 0.3f; } // starker Flip → deutlich dämpfen
    }
    g_zlock.vx = 0.7f*g_zlock.vx + 0.3f*dirx;
    g_zlock.vy = 0.7f*g_zlock.vy + 0.3f*diry;

    // Grobe Schrittgröße (exakt in evaluateAndApply)
    const float invZ=1.0f/std::max(1e-6f,zoom);
    const float step=clampf(kSEED_STEP_NDC,0.f,kSTEP_MAX_NDC);

    out.shouldZoom = true;
    out.bestIndex  = g_zlock.lockIndex;
    out.isNewTarget= true;
    out.newOffsetX = previousOffset.x + dirx*(step*invZ);
    out.newOffsetY = previousOffset.y + diry*(step*invZ);
    out.distance   = step*invZ;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][TARGET] lock=%d ndc=(%.3f,%.3f) dir=(%.3f,%.3f) score=%.3f thr=%.3f",
                       g_zlock.lockIndex,(double)g_zlock.lockNdcX,(double)g_zlock.lockNdcY,
                       (double)dirx,(double)diry,(double)g_zlock.lockScore,(double)thr);
    }
    return out;
}

void evaluateAndApply(::FrameContext& fctx,
                      ::RendererState& state,
                      ZoomState& /*bus*/,
                      float /*gain*/) noexcept
{
    zs_beginFrame();

    if (CudaInterop::getPauseZoom()) {
        fctx.shouldZoom = false;
        fctx.newOffset  = fctx.offset;
        fctx.newOffsetD = { (double)fctx.offset.x, (double)fctx.offset.y };
        return;
    }

    // Overlay-Grid
    const int overlayTilePx = std::max(1, (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
    const int tilesY = (fctx.height + overlayTilePx - 1) / overlayTilePx;

    const float2 prevF = fctx.offset;

    ZoomResult zr = evaluateZoomTarget(
        state.h_entropy, state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,
        prevF,
        *(ZoomState*)nullptr // ungenutzt; nur Signatur
    );

    if (!zr.shouldZoom) {
        fctx.shouldZoom = false;
        fctx.newOffset  = prevF;
        fctx.newOffsetD = { (double)prevF.x, (double)prevF.y };
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][EVAL] should=0 keep=(%.9f,%.9f)", (double)fctx.newOffset.x, (double)fctx.newOffset.y);
        }
        return;
    }

    // --- NDC → Welt (double-präzise) ---
    double stepX=0.0, stepY=0.0;
    capy_pixel_steps_from_zoom_scale(
        (double)state.pixelScale.x, (double)state.pixelScale.y,
        fctx.width, fctx.zoomD > 0.0 ? fctx.zoomD : (double)state.zoom,
        stepX, stepY
    );

    // Richtung (normiert) aus zr (prev → proposed)
    double dirx = (double)(zr.newOffsetX - prevF.x);
    double diry = (double)(zr.newOffsetY - prevF.y);
    { const double n2 = dirx*dirx + diry*diry; if(!(n2>0.0)){ dirx=1.0; diry=0.0; } else { const double inv=1.0/std::sqrt(n2); dirx*=inv; diry*=inv; } }

    const double halfW = 0.5 * (double)fctx.width  * std::fabs(stepX);
    const double halfH = 0.5 * (double)fctx.height * std::fabs(stepY);

    float ndcTX=0.f, ndcTY=0.f;
    if (zr.bestIndex >= 0) tileIndexToNdcCenter(tilesX, tilesY, zr.bestIndex, ndcTX, ndcTY);
    const float distNdc = std::sqrt(ndcTX*ndcTX + ndcTY*ndcTY);
    const double stepNdc = (double)clampf(kSEED_STEP_NDC * (0.6f + 0.8f * distNdc), 0.6f * kSEED_STEP_NDC, kSTEP_MAX_NDC);

    double moveX = dirx * (stepNdc * halfW);
    double moveY = diry * (stepNdc * halfH);

    // Min. 1/2 Pixel-Schritt
    const double minPixStep = std::min(std::fabs(stepX), std::fabs(stepY));
    const double movLen = std::hypot(moveX, moveY);
    if (movLen < 0.5 * minPixStep && movLen > 0.0) { const double s = (0.5 * minPixStep) / movLen; moveX *= s; moveY *= s; }
    else if (!(movLen > 0.0)) { moveX = minPixStep; moveY = 0.0; }

    // *** Pixel-Move-Cap (zusätzlicher Beruhiger) ***
    {
        const double capX = kMAX_PX_MOVE_PER_FRAME * std::fabs(stepX);
        const double capY = kMAX_PX_MOVE_PER_FRAME * std::fabs(stepY);
        if (std::fabs(moveX) > capX) moveX = (moveX > 0 ? capX : -capX);
        if (std::fabs(moveY) > capY) moveY = (moveY > 0 ? capY : -capY);
    }

    // In-Set-Veto (Cardioid/Bulb) → ggf. Nachbar mit max edgeScore
    if (zr.bestIndex >= 0) {
        const int bx = zr.bestIndex % tilesX, by = zr.bestIndex / tilesX;
        const double px = (double)ndcTX * 0.5 * (double)fctx.width;
        const double py = (double)ndcTY * 0.5 * (double)fctx.height;
        const double wx = (double)state.center.x + px * stepX;
        const double wy = (double)state.center.y + py * stepY;

        if (insideCardioidOrBulb(wx, wy)) {
            int bestN=-1; float bestS=-1e30f; float ndcNX=0.f, ndcNY=0.f;
            const int nx4[4]={bx-1,bx+1,bx,  bx}; const int ny4[4]={by,  by,  by-1,by+1};
            for(int k=0;k<4;++k){
                const int xn=nx4[k], yn=ny4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
                const float s = edgeScoreAt(state.h_contrast,xn,yn,tilesX,tilesY);
                if (s > bestS){ bestS=s; bestN=idxAt(xn,yn,tilesX); tileIndexToNdcCenter(tilesX,tilesY,bestN,ndcNX,ndcNY); }
            }
            if (bestN >= 0) {
                double dx=(double)ndcNX, dy=(double)ndcNY; const double n2=dx*dx+dy*dy;
                if (n2>0.0){ const double inv=1.0/std::sqrt(n2); dx*=inv; dy*=inv; }
                moveX = dx * (stepNdc * halfW);
                moveY = dy * (stepNdc * halfH);
            }
        }
    }

    // Blend auf Center (sanft)
    const double baseX = (double)state.center.x;
    const double baseY = (double)state.center.y;
    const double txD = baseX * (1.0 - (double)kBLEND_A) + (baseX + moveX) * (double)kBLEND_A;
    const double tyD = baseY * (1.0 - (double)kBLEND_A) + (baseY + moveY) * (double)kBLEND_A;

    fctx.shouldZoom = true;
    fctx.newOffsetD = { txD, tyD };
    fctx.newOffset  = make_float2((float)txD, (float)tyD);

    if constexpr (Settings::debugLogging) {
        const double ulpX = std::nextafter(baseX, baseX + 1.0) - baseX;
        const double ulpY = std::nextafter(baseY, baseY + 1.0) - baseY;
        LUCHS_LOG_HOST("[ZOOM][APPLY] i=%d ndc=(%.3f,%.3f) move=(%.3e,%.3e) minPx=%.3e ulp=(%.3e,%.3e) new=(%.9f,%.9f)",
                       zr.bestIndex, (double)ndcTX, (double)ndcTY, moveX, moveY, minPixStep, ulpX, ulpY, txD, tyD);
    }
}

} // namespace ZoomLogic
