///// Otter: Silk-Lite auto-pan/zoom controller (lokal, kantengetrieben, ohne Heatmap-Zwang).
///// Schneefuchs: Lock + Cooldown + Hysterese, harte Entzerrung von Richtungswechseln.
///// Maus: Ziel = nächstgelegene valide Kachel; Richtung = lokaler Kontrastgradient; ASCII-only Logs.
///// Datei: src/zoom_logic.cpp

#include "zoom_logic.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "capybara_mapping.cuh"

#include <algorithm>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

namespace {

// ----------------------------- Tunables (kurz & wirksam) -----------------
constexpr float kSeedStepNDC       = 0.012f;  // Grundschritt (NDC-Anteil der Halbbreite/-höhe)
constexpr float kStepMaxNDC        = 0.28f;   // Sicherheitskappe
constexpr float kBlendAlpha        = 0.14f;   // sanfter Center-Blend
constexpr int   kSearchR           = 2;       // lokales Kachelfenster
constexpr float kThrMedAddMAD      = 0.5f;    // Schwelle = median + a*MAD

// Stabilisierung
constexpr int   kCooldownFrames    = 10;
constexpr float kHystFactor        = 1.25f;   // „besser“-Faktor
constexpr float kMuchBetter        = 1.50f;   // darf Cooldown brechen
constexpr float kMinRetargetNDC    = 0.06f;   // min. Abstand alt→neu (NDC)
constexpr float kLockBoxNDC        = 0.25f;   // harte Lock-Box (nahe Ziele ignorieren)
constexpr int   kRetargetStreak    = 4;       // gleicher Kandidat N-mal in Folge nötig
constexpr float kMaxTurnDeg        = 12.0f;   // max Winkeländerung/Frame
constexpr float kDirEmaKeep        = 0.85f;   // Richtungs-EMA (1-β)
constexpr double kMaxPixStep       = 0.40;    // max Bewegung in Pixeln/Frame (Welt)

// ----------------------------- helpers -----------------------------------
inline float clampf(float x, float a, float b){ return (x<a)?a:((x>b)?b:x); }
inline bool normalize2D(float& x,float& y){ const float n2=x*x+y*y; if(!(n2>0.f))return false; const float inv=1.f/std::sqrt(n2); x*=inv; y*=inv; return true; }

float median_inplace(std::vector<float>& v){
    if (v.empty()) return 0.f;
    const size_t m=v.size()/2;
    std::nth_element(v.begin(), v.begin()+m, v.end());
    float med=v[m];
    if ((v.size()&1)==0){ std::nth_element(v.begin(), v.begin()+m-1, v.begin()+m); med = 0.5f*(med+v[m-1]); }
    return med;
}
float mad_from_center_inplace(std::vector<float>& v,float c){ for(auto& x:v) x=std::fabs(x-c); return std::max(1e-6f, median_inplace(v)); }

inline int idxAt(int x,int y,int tilesX){ return y*tilesX+x; }
inline void tileIndexToNdcCenter(int tilesX,int tilesY,int idx,float& ndcX,float& ndcY){
    const int tx = (tilesX>0)?(idx%tilesX):0, ty=(tilesX>0)?(idx/tilesX):0;
    const float cx=(tx+0.5f)/std::max(1,tilesX), cy=(ty+0.5f)/std::max(1,tilesY);
    ndcX = cx*2.f-1.f; ndcY = cy*2.f-1.f;
}

inline float edgeScoreAt(const std::vector<float>& c,int x,int y,int tx,int ty){
    const int i=idxAt(x,y,tx); const float lc=c[(size_t)i];
    float sum=0.f; int n=0;
    const int nx4[4]={x-1,x+1,x,  x}, ny4[4]={y,  y,  y-1,y+1};
    for(int k=0;k<4;++k){ const int xn=nx4[k], yn=ny4[k];
        if(xn<0||yn<0||xn>=tx||yn>=ty) continue;
        sum += std::fabs(c[(size_t)idxAt(xn,yn,tx)] - lc); ++n;
    }
    const float nd = (n>0)?(sum/(float)n):0.f;
    return lc + 0.75f*nd;
}

inline float angleDeg(float ax,float ay,float bx,float by){
    const float La2=ax*ax+ay*ay, Lb2=bx*bx+by*by; if(!(La2>0.f)||!(Lb2>0.f)) return 0.f;
    float c=(ax*bx+ay*by)/(std::sqrt(La2)*std::sqrt(Lb2)); c=clampf(c,-1.f,1.f);
    return std::acos(c)*57.29577951308232f;
}

// ----------------------------- tiny state --------------------------------
struct ZLock {
    bool  init=false;
    int   lock=-1;       // gelockter Index
    float score=0.f;
    float ndcX=0.f, ndcY=0.f;

    int   cooldown=0;
    int   streakCnt=0;   // wie oft derselbe Kandidat in Folge
    int   lastCand=-1;

    float vx=1.f, vy=0.f; // Richtungs-EMA (NDC)
};
static ZLock g;

inline void beginFrame(){ if(g.cooldown>0) g.cooldown--; g.vx*=0.92f; g.vy*=0.92f; }

// Hauptmengen-Test (Cardioid + Bulb) für Weltkoordinaten
inline bool insideCardioidOrBulb(double x,double y) noexcept {
    const double xm=x-0.25, q=xm*xm+y*y;
    if (q*(q+xm) < 0.25*y*y) return true;
    const double dx=x+1.0; return (dx*dx+y*y) < 0.0625;
}

} // namespace

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>& /*entropy*/,
                              const std::vector<float>& contrast,
                              int tilesX,int tilesY,
                              int /*width*/,int /*height*/,
                              float2 /*curOffset*/, float zoom,
                              float2 prevOffset,
                              ZoomState& /*unused*/) noexcept
{
    ZoomResult out{};

    const int total = (tilesX>0 && tilesY>0) ? tilesX*tilesY : 0;
    if (total<=0 || (int)contrast.size()<total){
        out.shouldZoom = Settings::ForceAlwaysZoom; return out;
    }

    // robuste Kontrastschwelle
    std::vector<float> tmp=contrast;
    const float med = median_inplace(tmp);
    const float mad = mad_from_center_inplace(tmp, med);
    const float thr = med + kThrMedAddMAD*mad;

    // bestes Ziel im Nahfenster um Bildmitte
    const int cx=tilesX/2, cy=tilesY/2;
    int   bestI=-1; float bestS=-1e30f, bestD2=1e30f, bestNX=0.f, bestNY=0.f;

    for(int dy=-kSearchR; dy<=kSearchR; ++dy){
        for(int dx=-kSearchR; dx<=kSearchR; ++dx){
            const int x=cx+dx, y=cy+dy; if(x<0||y<0||x>=tilesX||y>=tilesY) continue;
            const float s=edgeScoreAt(contrast,x,y,tilesX,tilesY); if(!(s>thr)) continue;
            float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(x,y,tilesX),nx,ny);
            const float d2=nx*nx+ny*ny;
            if (d2<bestD2-1e-6f || (std::fabs(d2-bestD2)<=1e-6f && s>bestS)){ bestD2=d2; bestS=s; bestI=idxAt(x,y,tilesX); bestNX=nx; bestNY=ny; }
        }
    }

    // kein Kandidat → sanfter Drift in gelockte Richtung
    if (bestI<0){
        out.shouldZoom = Settings::ForceAlwaysZoom;
        float dirx = g.init ? g.ndcX : 1.f, diry = g.init ? g.ndcY : 0.f;
        normalize2D(dirx,diry);
        const float invZ = 1.f/std::max(1e-6f, zoom);
        const float step = clampf(kSeedStepNDC, 0.f, kStepMaxNDC);
        out.newOffsetX = prevOffset.x + dirx*(step*invZ);
        out.newOffsetY = prevOffset.y + diry*(step*invZ);
        out.distance   = step*invZ;
        return out;
    }

    // Erstinitialisierung
    if(!g.init){
        g.init=true; g.lock=bestI; g.score=bestS; g.ndcX=bestNX; g.ndcY=bestNY; g.cooldown=kCooldownFrames; g.lastCand=bestI; g.streakCnt=1;
    } else {
        // streak-basierte Retarget-Entscheidung (entkoppelt von Zufallsflattern)
        g.streakCnt = (bestI==g.lastCand)? (g.streakCnt+1) : 1;
        g.lastCand  = bestI;

        float dX = bestNX-g.ndcX, dY = bestNY-g.ndcY;
        const float dist = std::sqrt(dX*dX+dY*dY);
        const bool farEnough  = dist >= kMinRetargetNDC;
        const bool better     = bestS >= g.score * kHystFactor;
        const bool muchBetter = bestS >= g.score * kMuchBetter;
        const bool inLockBox  = dist <= kLockBoxNDC;

        const bool allow = ( (g.cooldown<=0 && farEnough && better && g.streakCnt>=kRetargetStreak && !inLockBox)
                           || (muchBetter && farEnough) );

        if (allow){
            g.lock = bestI; g.score=bestS; g.ndcX=bestNX; g.ndcY=bestNY; g.cooldown=kCooldownFrames;
        }
    }

    // Richtungsvektor = lokaler Kontrastgradient um den Lock
    const int bx=g.lock%tilesX, by=g.lock/tilesX;
    auto ndcOf=[&](int X,int Y){ float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(X,Y,tilesX),nx,ny); return make_float2(nx,ny); };
    const float2 pC = ndcOf(bx,by);

    float2 grad=make_float2(0,0);
    const int nx4[4]={bx-1,bx+1,bx,  bx}, ny4[4]={by,  by,  by-1,by+1};
    for(int k=0;k<4;++k){
        const int xn=nx4[k], yn=ny4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        const float w = contrast[(size_t)idxAt(xn,yn,tilesX)] - contrast[(size_t)g.lock];
        const float2 pN=ndcOf(xn,yn);
        float vx=pN.x-pC.x, vy=pN.y-pC.y; if(normalize2D(vx,vy)){ grad.x+=w*vx; grad.y+=w*vy; }
    }
    float dirx=(std::fabs(grad.x)+std::fabs(grad.y)>0.f)? grad.x : g.ndcX;
    float diry=(std::fabs(grad.x)+std::fabs(grad.y)>0.f)? grad.y : g.ndcY;
    if(!normalize2D(dirx,diry)){ dirx=1.f; diry=0.f; }

    // Winkel-Limit + Richtungs-EMA
    if (g.vx*g.vx + g.vy*g.vy > 0.f){
        const float turn = angleDeg(g.vx,g.vy,dirx,diry);
        if (turn > kMaxTurnDeg){
            const float t = kMaxTurnDeg / turn;
            float ox = (1.f-t)*g.vx + t*dirx, oy = (1.f-t)*g.vy + t*diry; normalize2D(ox,oy);
            dirx=ox; diry=oy;
        }
    }
    g.vx = kDirEmaKeep*g.vx + (1.f-kDirEmaKeep)*dirx;
    g.vy = kDirEmaKeep*g.vy + (1.f-kDirEmaKeep)*diry;
    normalize2D(g.vx,g.vy);
    dirx=g.vx; diry=g.vy;

    // grobe NDC-Schrittweite (final in evaluateAndApply in Welt begrenzt)
    const float invZ = 1.f/std::max(1e-6f, zoom);
    const float step = clampf(kSeedStepNDC, 0.f, kStepMaxNDC);

    out.shouldZoom = true;
    out.bestIndex  = g.lock;
    out.isNewTarget= true;
    out.newOffsetX = prevOffset.x + dirx*(step*invZ);
    out.newOffsetY = prevOffset.y + diry*(step*invZ);
    out.distance   = step*invZ;
    return out;
}

void evaluateAndApply(::FrameContext& fctx,
                      ::RendererState& state,
                      ZoomState& /*bus*/,
                      float /*gain*/) noexcept
{
    beginFrame();

    if (CudaInterop::getPauseZoom()){
        fctx.shouldZoom=false; fctx.newOffset=fctx.offset; fctx.newOffsetD={ (double)fctx.offset.x,(double)fctx.offset.y }; return;
    }

    // Overlay-Grid (stabil gleich wie UI)
    const int overlayPx = std::max(1, (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + overlayPx - 1) / overlayPx;
    const int tilesY = (fctx.height + overlayPx - 1) / overlayPx;

    const float2 prevF = fctx.offset;

    ZoomResult zr = evaluateZoomTarget(
        state.h_entropy, state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,
        prevF, *(ZoomState*)nullptr
    );

    if (!zr.shouldZoom){
        fctx.shouldZoom=false; fctx.newOffset=prevF; fctx.newOffsetD={ (double)prevF.x,(double)prevF.y }; return;
    }

    // --- NDC → Welt (double) ---
    double stepX=0.0, stepY=0.0;
    capy_pixel_steps_from_zoom_scale(
        (double)state.pixelScale.x,(double)state.pixelScale.y,
        fctx.width, fctx.zoomD>0.0? fctx.zoomD : (double)state.zoom,
        stepX, stepY
    );

    // Richtung aus zr (normiert)
    double dx = (double)(zr.newOffsetX - prevF.x);
    double dy = (double)(zr.newOffsetY - prevF.y);
    { const double n2=dx*dx+dy*dy; if(!(n2>0.0)){ dx=1.0; dy=0.0; } else { const double inv=1.0/std::sqrt(n2); dx*=inv; dy*=inv; } }

    // Basis-Schritt in Welt
    const double halfW = 0.5 * (double)fctx.width  * std::fabs(stepX);
    const double halfH = 0.5 * (double)fctx.height * std::fabs(stepY);
    const double stepNdc = (double)clampf(kSeedStepNDC, 0.0f, kStepMaxNDC);
    double moveX = dx * (stepNdc * halfW);
    double moveY = dy * (stepNdc * halfH);

    // Pixel-Kappung (harte Bremse gegen Flattern)
    const double pixLen = std::hypot(moveX/std::max(1e-16,std::fabs(stepX)),
                                     moveY/std::max(1e-16,std::fabs(stepY)));
    if (pixLen > kMaxPixStep && pixLen > 0.0){
        const double s = kMaxPixStep / pixLen;
        moveX *= s; moveY *= s;
    }

    // In-Set-Veto (Lock-Kachelzentrum in Mandelbrot-Hauptmenge?) → nimm besten Nachbarn
    if (zr.bestIndex>=0){
        float ndcX=0.f, ndcY=0.f; tileIndexToNdcCenter(tilesX,tilesY,zr.bestIndex,ndcX,ndcY);
        const double px = (double)ndcX * 0.5 * (double)fctx.width;
        const double py = (double)ndcY * 0.5 * (double)fctx.height;
        const double wx = (double)state.center.x + px*stepX;
        const double wy = (double)state.center.y + py*stepY;
        if (insideCardioidOrBulb(wx,wy)){
            const int bx=zr.bestIndex%tilesX, by=zr.bestIndex/tilesX;
            int bestN=-1; float bestS=-1e30f; float nx=0.f, ny=0.f;
            const int nx4[4]={bx-1,bx+1,bx,  bx}, ny4[4]={by,  by,  by-1,by+1};
            for(int k=0;k<4;++k){
                const int xn=nx4[k], yn=ny4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
                const float s=edgeScoreAt(state.h_contrast,xn,yn,tilesX,tilesY);
                if(s>bestS){ bestS=s; bestN=idxAt(xn,yn,tilesX); tileIndexToNdcCenter(tilesX,tilesY,bestN,nx,ny); }
            }
            if(bestN>=0){
                double ddx=nx, ddy=ny; const double n2=ddx*ddx+ddy*ddy;
                if(n2>0.0){ const double inv=1.0/std::sqrt(n2); ddx*=inv; ddy*=inv; }
                moveX = ddx*(stepNdc*halfW); moveY = ddy*(stepNdc*halfH);
            }
        }
    }

    // sanfter Blend
    const double bx = (double)state.center.x, by=(double)state.center.y;
    const double tx = bx*(1.0-(double)kBlendAlpha) + (bx+moveX)*(double)kBlendAlpha;
    const double ty = by*(1.0-(double)kBlendAlpha) + (by+moveY)*(double)kBlendAlpha;

    fctx.shouldZoom = true;
    fctx.newOffsetD = { tx, ty };
    fctx.newOffset  = make_float2((float)tx,(float)ty);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][APPLY] lock=%d movePix=%.3f new=(%.9f,%.9f)",
            zr.bestIndex, std::hypot(moveX/std::max(1e-16,std::fabs(stepX)),
                                     moveY/std::max(1e-16,std::fabs(stepY))),
            tx, ty);
    }
}

} // namespace ZoomLogic
