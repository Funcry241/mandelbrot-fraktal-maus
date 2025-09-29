///// Otter: auto-pan/zoom (lokal, kantengetrieben).
///// Schneefuchs: Anti-Flip, adaptiver Turn-Boost, schnellere EMA, reduziertes Blend bei großen Drehungen.
///@@@ Maus: Ziel = nächste valide Kachel; Richtung = lokaler Kontrastgradient.
// Datei: src/zoom_logic.cpp

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

namespace {

// ---- Tunables (responsiver) ----
constexpr float  kSeedStepNDC   = 0.012f;
constexpr float  kStepMaxNDC    = 0.28f;
constexpr float  kBlendABase    = 0.14f;  // Grund-Blend
constexpr int    kSearchR       = 2;
constexpr int    kCooldownFrames= 8;      // etwas kürzer
constexpr int    kRetargetStreak= 2;
constexpr float  kHyst          = 1.15f;
constexpr float  kMuch          = 1.30f;
constexpr float  kMinRetarget   = 0.08f;
constexpr float  kLockBox       = 0.15f;  // kleiner, aggressiver
constexpr float  kMaxTurnDeg    = 35.0f;  // deutlich höher = snappier
constexpr float  kThrMedAddMAD  = 0.50f;

constexpr double kTurnBoostDeg  = 55.0;   // ab diesem Winkel einmaliger Boost …
constexpr double kTurnBoostMul  = 1.35;   // … um den Schritt zu beschleunigen

inline float  clampf(float x,float a,float b){return x<a?a:(x>b?b:x);}
inline bool   norm2(float& x,float& y){float n=x*x+y*y; if(!(n>0))return false; float inv=1/std::sqrt(n); x*=inv; y*=inv; return true;}

float median_inplace(std::vector<float>& v){
    if(v.empty()) return 0.f; size_t m=v.size()/2;
    std::nth_element(v.begin(),v.begin()+m,v.end()); float med=v[m];
    if(!(v.size()&1)){ std::nth_element(v.begin(),v.begin()+m-1,v.begin()+m); med=0.5f*(med+v[m-1]); }
    return med;
}
float mad_from_center_inplace(std::vector<float>& v,float c){ for(auto& x:v) x=std::fabs(x-c); float mad=median_inplace(v); return (mad>1e-6f)?mad:1.f; }

inline int   idxAt(int x,int y,int sx){ return y*sx+x; }
inline void  ndcCenter(int sx,int sy,int idx,float& nx,float& ny){
    int tx=(sx>0)?idx%sx:0, ty=(sx>0)?idx/sx:0;
    float cx=(tx+0.5f)/std::max(1,sx), cy=(ty+0.5f)/std::max(1,sy);
    nx=cx*2.f-1.f; ny=cy*2.f-1.f;
}
inline float edgeScoreAt(const std::vector<float>& c,int x,int y,int sx,int sy){
    int i=idxAt(x,y,sx); float lc=c[(size_t)i], acc=0.f; int n=0;
    const int dx[4]={-1,1,0,0}, dy[4]={0,0,-1,1};
    for(int k=0;k<4;++k){ int xn=x+dx[k], yn=y+dy[k]; if(xn<0||yn<0||xn>=sx||yn>=sy) continue; acc+=std::fabs(c[(size_t)idxAt(xn,yn,sx)]-lc); ++n; }
    return lc + 0.75f*((n>0)?acc/n:0.f);
}
inline float angleDeg(float ax,float ay,float bx,float by){
    float a2=ax*ax+ay*ay, b2=bx*bx+by*by; if(!(a2>0.f)||!(b2>0.f)) return 0.f;
    float c=(ax*bx+ay*by)/(std::sqrt(a2)*std::sqrt(b2)); c=clampf(c,-1.f,1.f);
    return std::acos(c)*57.29577951308232f;
}
inline bool insideCardioidOrBulb(double x,double y) noexcept{
    const double xm=x-0.25, q=xm*xm+y*y; if(q*(q+xm)<0.25*y*y) return true;
    const double dx=x+1.0; return (dx*dx+y*y)<0.0625;
}

// ---- Minimaler Zustand ----
struct ZLock{
    bool  init=false;
    int   lock=-1,lastCand=-1,streak=0,cooldown=0;
    float score=0, nx=1, ny=0, vx=1, vy=0;
};
static ZLock g;

// für adaptive Apply-Phase
static double g_lastTurnDeg = 0.0;

inline void beginFrame(){
    if(g.cooldown>0)--g.cooldown;
    // weniger Dämpfung, damit Richtungswechsel schneller „durchkommen“
    g.vx*=0.97f; g.vy*=0.97f;
}

} // anon

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>&/*entropy*/,
                              const std::vector<float>& contrast,
                              int tilesX,int tilesY,
                              int/*w*/,int/*h*/,
                              float2/*cur*/, float zoom,
                              float2 prev, ZoomState&/*unused*/) noexcept
{
    ZoomResult out{};
    g_lastTurnDeg = 0.0; // default

    const int total=(tilesX>0&&tilesY>0)?tilesX*tilesY:0;
    if(total<=0 || (int)contrast.size()<total){
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZOOM][WARN] contrast invalid/empty, fallback drift");
        out.shouldZoom=Settings::ForceAlwaysZoom;
        return out;
    }

    std::vector<float> tmp=contrast;
    float med=median_inplace(tmp), mad=mad_from_center_inplace(tmp,med);
    const float thr = med + kThrMedAddMAD*mad;

    // Kandidat um Bildmitte
    const int cx=tilesX/2, cy=tilesY/2;
    int bestI=-1; float bestS=-1e30f, bestD2=1e30f, bnx=0,bny=0;
    for(int dy=-kSearchR; dy<=kSearchR; ++dy) for(int dx=-kSearchR; dx<=kSearchR; ++dx){
        int x=cx+dx, y=cy+dy; if(x<0||y<0||x>=tilesX||y>=tilesY) continue;
        float s=edgeScoreAt(contrast,x,y,tilesX,tilesY); if(!(s>thr)) continue;
        float nx,ny; ndcCenter(tilesX,tilesY,idxAt(x,y,tilesX),nx,ny); float d2=nx*nx+ny*ny;
        if(d2<bestD2-1e-6f || (std::fabs(d2-bestD2)<=1e-6f && s>bestS)){ bestI=idxAt(x,y,tilesX); bestS=s; bestD2=d2; bnx=nx; bny=ny; }
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZOOM][CAND] i=%d s=%.3f thr=%.3f nx=%.2f ny=%.2f", idxAt(x,y,tilesX), s, thr, nx, ny);
    }

    // Initialisierung / kein Kandidat
    if(!g.init && bestI>=0){
        g.init=true; g.lock=bestI; g.score=bestS; g.nx=bnx; g.ny=bny; g.vx=bnx; g.vy=bny;
        g.cooldown=kCooldownFrames; g.lastCand=bestI; g.streak=1;
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZOOM][LOCK][INIT] i=%d score=%.3f", g.lock, g.score);
    }
    if(bestI<0){
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZOOM][FALLBACK] no candidate, drift forward");
        float dx=g.nx, dy=g.ny; norm2(dx,dy);
        float invZ=1.f/std::max(1e-6f,zoom), step=clampf(kSeedStepNDC,0.f,kStepMaxNDC);
        out.shouldZoom=true; out.newOffsetX=prev.x+dx*(step*invZ); out.newOffsetY=prev.y+dy*(step*invZ); out.distance=step*invZ; return out;
    }

    // Streak/Hysterese/Cooldown/LockBox
    g.streak = (bestI==g.lastCand)? (g.streak+1) : 1; g.lastCand=bestI;
    float dnx=bnx-g.nx, dny=bny-g.ny; float dist=std::sqrt(dnx*dnx+dny*dny);
    bool far = dist>=kMinRetarget, better=bestS>=g.score*kHyst, much=bestS>=g.score*kMuch, inBox=dist<=kLockBox;
    if(g.init && ((g.cooldown<=0 && far && better && g.streak>=kRetargetStreak && !inBox) || (much && far))){
        g.lock=bestI; g.score=bestS; g.nx=bnx; g.ny=bny; g.cooldown=kCooldownFrames;
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZOOM][LOCK][RETARGET] i=%d score=%.3f streak=%d", g.lock, g.score, g.streak);
    }

    // Richtung = lokaler Gradientenstoß (4-Nachbarn)
    const int bx=g.lock%tilesX, by=g.lock/tilesX;
    auto ndcOf=[&](int X,int Y){ float nx,ny; ndcCenter(tilesX,tilesY,idxAt(X,Y,tilesX),nx,ny); return std::pair<float,float>{nx,ny}; };
    float cxN,cyN; std::tie(cxN,cyN)=ndcOf(bx,by);
    float gx=0,gy=0; const int dx4[4]={-1,1,0,0}, dy4[4]={0,0,-1,1};
    for(int k=0;k<4;++k){
        int xn=bx+dx4[k], yn=by+dy4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        float w=contrast[(size_t)idxAt(xn,yn,tilesX)]-contrast[(size_t)g.lock];
        float nx,ny; std::tie(nx,ny)=ndcOf(xn,yn); float vx=nx-cxN, vy=ny-cyN; if(norm2(vx,vy)){ gx+=w*vx; gy+=w*vy; }
    }
    float dirx=(std::fabs(gx)+std::fabs(gy)>0)?gx:g.nx, diry=(std::fabs(gx)+std::fabs(gy)>0)?gy:g.ny; norm2(dirx,diry);

    // ---- Anti-Flip + großzügiger Turn-Clamp ----
    if(std::fabs(g.vx)+std::fabs(g.vy)>0.f){
        const float pdot = g.vx*dirx + g.vy*diry;
        // Keine harte Umkehr mehr, nur abfedern
        if(pdot < 0.0f){ dirx = 0.6f*dirx + 0.4f*g.vx; diry = 0.6f*diry + 0.4f*g.vy; norm2(dirx,diry); }
        const float turn=angleDeg(g.vx,g.vy,dirx,diry);
        g_lastTurnDeg = turn; // für Apply-Phase
        if(turn>1e-3f){
            const float t=clampf(kMaxTurnDeg/turn,0.f,1.f);
            float ox=(1.f-t)*g.vx + t*dirx, oy=(1.f-t)*g.vy + t*diry; norm2(ox,oy);
            dirx=ox; diry=oy;
        }
    } else {
        g_lastTurnDeg = 0.0;
    }

    // Schnellere Richtungs-EMA (snappier)
    g.vx = 0.75f*g.vx + 0.25f*dirx; g.vy = 0.75f*g.vy + 0.25f*diry; norm2(g.vx,g.vy);
    dirx=g.vx; diry=g.vy;

    float invZ=1.f/std::max(1e-6f,zoom), step=clampf(kSeedStepNDC,0.f,kStepMaxNDC);
    out.shouldZoom=true; out.bestIndex=g.lock; out.isNewTarget=true;
    out.newOffsetX=prev.x+dirx*(step*invZ); out.newOffsetY=prev.y+diry*(step*invZ); out.distance=step*invZ;

    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[ZOOM][TARGET] lock=%d dir=(%.2f,%.2f) step=%.4f turn=%.1f",
                       g.lock, dirx, diry, step*invZ, g_lastTurnDeg);

    return out;
}

void evaluateAndApply(::FrameContext& fctx, ::RendererState& state, ZoomState& bus, float/*gain*/) noexcept
{
    beginFrame();
    bus.hadCandidate = false;

    if (CudaInterop::getPauseZoom()) {
        fctx.shouldZoom = false;
        fctx.newOffset  = fctx.offset;
        fctx.newOffsetD = { (double)fctx.offset.x, (double)fctx.offset.y };
        return;
    }

    const int tilePx = std::max(1, (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + tilePx - 1) / tilePx;
    const int tilesY = (fctx.height + tilePx - 1) / tilePx;

    const float2 prevF = fctx.offset;
    ZoomResult zr = evaluateZoomTarget(
        state.h_entropy, state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,
        prevF, *(ZoomState*)nullptr
    );

    if (!zr.shouldZoom) {
        fctx.shouldZoom = false;
        fctx.newOffset  = prevF;
        fctx.newOffsetD = { (double)prevF.x, (double)prevF.y };
        return;
    }

    bus.hadCandidate = (zr.bestIndex >= 0);

    // NDC -> Welt
    double stepX = 0, stepY = 0;
    capy_pixel_steps_from_zoom_scale(
        (double)state.pixelScale.x, (double)state.pixelScale.y,
        fctx.width, (fctx.zoomD > 0.0 ? fctx.zoomD : (double)state.zoom),
        stepX, stepY
    );

    double dx = (double)(zr.newOffsetX - prevF.x);
    double dy = (double)(zr.newOffsetY - prevF.y);
    {
        const double n2 = dx*dx + dy*dy;
        if (!(n2 > 0.0)) { dx = 1.0; dy = 0.0; }
        else { const double inv = 1.0 / std::sqrt(n2); dx *= inv; dy *= inv; }
    }

    const double halfW = 0.5 * (double)fctx.width  * std::fabs(stepX);
    const double halfH = 0.5 * (double)fctx.height * std::fabs(stepY);

    double stepNdc = (double)clampf(kSeedStepNDC, 0.0f, kStepMaxNDC);

    // Adaptiver Turn-Boost: bei großen Drehungen kurz kräftiger treten
    if (g_lastTurnDeg > kTurnBoostDeg) {
        // *** FIX: expliziter Typ, vermeidet MSVC-Ambiguität (float vs. double) ***
        stepNdc = std::min<double>(stepNdc * kTurnBoostMul, static_cast<double>(kStepMaxNDC));
    }

    double moveX = dx * (stepNdc * halfW);
    double moveY = dy * (stepNdc * halfH);

    // min. 0.5 Pixel
    const double minPix = std::min(std::fabs(stepX), std::fabs(stepY));
    const double len    = std::hypot(moveX, moveY);
    if (len < 0.5 * minPix && len > 0.0) {
        const double s = (0.5 * minPix) / len;
        moveX *= s; moveY *= s;
    }

    // In-Set-Veto: bremsen
    if (zr.bestIndex >= 0) {
        float nx = 0, ny = 0;
        ndcCenter(tilesX, tilesY, zr.bestIndex, nx, ny);
        const double px = (double)nx * 0.5 * (double)fctx.width;
        const double py = (double)ny * 0.5 * (double)fctx.height;
        const double wx = (double)state.center.x + px * stepX;
        const double wy = (double)state.center.y + py * stepY;
        if (insideCardioidOrBulb(wx, wy)) { moveX *= 0.6; moveY *= 0.6; }
    }

    // Blend: bei großen Drehungen etwas weniger filtern
    const double blend = (g_lastTurnDeg > 45.0 ? kBlendABase * 0.65 : kBlendABase);

    const double baseX = (double)state.center.x;
    const double baseY = (double)state.center.y;
    const double txD   = baseX * (1.0 - blend) + (baseX + moveX) * blend;
    const double tyD   = baseY * (1.0 - blend) + (baseY + moveY) * blend;

    fctx.shouldZoom = true;
    fctx.newOffsetD = { txD, tyD };
    fctx.newOffset  = make_float2((float)txD, (float)tyD);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][APPLY] turn=%.1f blend=%.3f move=(%.3e,%.3e) new=(%.9f,%.9f)",
                       g_lastTurnDeg, blend, moveX, moveY, txD, tyD);
    }
}

} // namespace ZoomLogic
