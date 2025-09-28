///// Otter: Silk-Lite auto-pan/zoom controller (lokal, kantengetrieben, ohne Heatmap-Zwang).
///// Schneefuchs: Lock+Cooldown+Hysterese, Sticky-Box, starker Turn-Limiter, leichte EMA.
///@@@ Maus: Ziel = nächste valide Kachel; Richtung = lokaler Kontrastgradient; ASCII-only Logs.
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

// ---------- TUNABLES (kurz & hart) ----------
constexpr float kSEED_STEP_NDC = 0.010f; // kleiner Basisschritt
constexpr float kSTEP_MAX_NDC  = 0.08f;  // enge Kappe
constexpr float kBLEND_A       = 0.14f;  // leichte EMA
constexpr int   kSEARCH_R      = 1;      // tiny Nachbarschaft
constexpr float kTHR_MED_ADD_MAD = 0.4f; // robustes Gate

// Retarget/Glättung
constexpr int   kCOOLDOWN_FRAMES     = 18;
constexpr float kHYST_FACTOR         = 1.35f; // deutlich besser
constexpr float kMUCH_BETTER_FACTOR  = 1.70f; // darf alles brechen
constexpr float kLOCKBOX_NDC         = 0.12f; // Sticky-Box
constexpr float kMIN_RETARGET_NDC    = 0.08f; // Mindestbewegung
constexpr float kMAX_TURN_DEG        = 12.0f; // pro Frame
constexpr float kHARD_FLIP_DEG       = 90.0f; // >90°: Step stark dämpfen
constexpr float kDIR_MOMENTUM        = 0.65f; // drift-bias

inline float clampf(float x, float a, float b){ return x<a?a:(x>b?b:x); }
inline bool norm2(float& x,float& y){ const float n2=x*x+y*y; if(!(n2>0))return false; const float inv=1/std::sqrt(n2); x*=inv; y*=inv; return true; }
inline float angleDeg(float ax,float ay,float bx,float by){
    if(!(ax||ay||bx||by)) return 0.f;
    float La=std::sqrt(ax*ax+ay*ay), Lb=std::sqrt(bx*bx+by*by);
    float c=(ax*bx+ay*by)/(La*Lb); c=clampf(c,-1.f,1.f); return std::acos(c)*57.2957795f;
}

inline void tileIndexToNdcCenter(int tilesX,int tilesY,int idx,float& x,float& y){
    const int tx=(tilesX>0)?(idx%tilesX):0, ty=(tilesX>0)?(idx/tilesX):0;
    const float cx=(tx+0.5f)/std::max(1,tilesX), cy=(ty+0.5f)/std::max(1,tilesY);
    x=cx*2.f-1.f; y=cy*2.f-1.f;
}
inline int idxAt(int x,int y,int sx){ return y*sx+x; }

float median_inplace(std::vector<float>& v){
    if(v.empty()) return 0.f; size_t m=v.size()/2;
    std::nth_element(v.begin(),v.begin()+m,v.end()); float med=v[m];
    if((v.size()&1)==0){ std::nth_element(v.begin(),v.begin()+m-1,v.begin()+m); med=0.5f*(med+v[m-1]); }
    return med;
}
float mad_from_center_inplace(std::vector<float>& v,float c){
    for(auto& x:v) x=std::fabs(x-c); float mad=median_inplace(v); return (mad>1e-6f)?mad:1.f;
}

// einfacher Kanten-Score (lokaler Kontrast + Nachbarsprung)
inline float edgeScoreAt(const std::vector<float>& c,int x,int y,int sx,int sy){
    const int i=idxAt(x,y,sx); const float lc=c[(size_t)i];
    float acc=0.f; int n=0; const int dx[4]={-1,1,0,0}, dy[4]={0,0,-1,1};
    for(int k=0;k<4;++k){ int xn=x+dx[k], yn=y+dy[k];
        if(xn<0||yn<0||xn>=sx||yn>=sy) continue;
        acc += std::fabs(c[(size_t)idxAt(xn,yn,sx)]-lc); ++n;
    }
    return lc + 0.75f*((n>0)?acc/n:0.f);
}

// Sticky-Lock
struct ZLock{
    bool init=false; int idx=-1; float score=0.f; float ndcX=0.f, ndcY=0.f; int cooldown=0;
    float vx=0.f, vy=0.f; // „Velocity“ in NDC (nur Richtung)
} static g;

inline void beginFrame(){ if(g.cooldown>0) --g.cooldown; g.vx*=0.9f; g.vy*=0.9f; }

// Mandelbrot-Hauptmenge (Cardioid+Bulb) – für simples In-Set-Veto
inline bool insideCardioidOrBulb(double x,double y) noexcept{
    const double xm=x-0.25, q=xm*xm+y*y; if(q*(q+xm)<0.25*y*y) return true;
    const double dx=x+1.0; return (dx*dx+y*y)<0.0625;
}

} // anon ns

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>& /*entropy*/,
                              const std::vector<float>& contrast,
                              int tilesX,int tilesY,
                              int /*w*/,int /*h*/,
                              float2 /*curOffset*/, float zoom,
                              float2 prevOff,
                              ZoomState& /*state*/) noexcept
{
    ZoomResult out{};
    const int total = (tilesX>0 && tilesY>0)? tilesX*tilesY : 0;
    if(total<=0 || (int)contrast.size()<total){ out.shouldZoom=Settings::ForceAlwaysZoom; return out; }

    // robustes Threshold
    std::vector<float> tmp = contrast;
    const float med = median_inplace(tmp);
    const float mad = mad_from_center_inplace(tmp, med);
    const float thr = med + kTHR_MED_ADD_MAD*mad;

    // Kandidat nahe Bildmitte
    const int cx=tilesX/2, cy=tilesY/2;
    int bestI=-1; float bestS=-1e30f, bestR2=1e30f, bnx=0, bny=0;
    for(int dy=-kSEARCH_R; dy<=kSEARCH_R; ++dy)
    for(int dx=-kSEARCH_R; dx<=kSEARCH_R; ++dx){
        int x=cx+dx, y=cy+dy; if(x<0||y<0||x>=tilesX||y>=tilesY) continue;
        float s=edgeScoreAt(contrast,x,y,tilesX,tilesY); if(!(s>thr)) continue;
        float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(x,y,tilesX),nx,ny);
        float r2=nx*nx+ny*ny;
        if(r2<bestR2-1e-6f || (std::fabs(r2-bestR2)<=1e-6f && s>bestS)){ bestI=idxAt(x,y,tilesX); bestS=s; bestR2=r2; bnx=nx; bny=ny; }
    }

    // Erstlock
    if(!g.init && bestI>=0){ g.init=true; g.idx=bestI; g.score=bestS; g.ndcX=bnx; g.ndcY=bny; g.cooldown=kCOOLDOWN_FRAMES; }

    // Fallback: keine Kante -> sanft entlang letzter Richtung
    if(bestI<0){
        float dx = g.init? g.ndcX : 1.f, dy = g.init? g.ndcY : 0.f; norm2(dx,dy);
        const float step = clampf(kSEED_STEP_NDC,0.f,kSTEP_MAX_NDC) / std::max(1e-6f, zoom);
        out.shouldZoom=true; out.newOffsetX=prevOff.x+dx*step; out.newOffsetY=prevOff.y+dy*step; out.distance=step;
        return out;
    }

    // Retarget-Logik (Sticky-Box + Hysterese + Cooldown)
    float dnx=bnx-g.ndcX, dny=bny-g.ndcY; const float d=std::sqrt(dnx*dnx+dny*dny);
    const bool farEnough = d >= kMIN_RETARGET_NDC;
    const bool inSticky  = d <= kLOCKBOX_NDC;
    const bool better    = bestS >= g.score * kHYST_FACTOR;
    const bool muchBetter= bestS >= g.score * kMUCH_BETTER_FACTOR;

    if(g.init){
        if( ((g.cooldown<=0)&&farEnough&&better&&!inSticky) || muchBetter ){
            g.idx=bestI; g.score=bestS; g.ndcX=bnx; g.ndcY=bny; g.cooldown=kCOOLDOWN_FRAMES;
        }
    }

    // Richtung: lokaler Gradientenstoß am Lock (oder zum Kachelzentrum)
    const int bx=g.idx%tilesX, by=g.idx/tilesX;
    auto ndcOf=[&](int X,int Y){ float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(X,Y,tilesX),nx,ny); return std::pair<float,float>{nx,ny}; };
    float cxN, cyN; std::tie(cxN,cyN)=ndcOf(bx,by);

    float gx=0, gy=0;
    const int dx4[4]={-1,1,0,0}, dy4[4]={0,0,-1,1};
    for(int k=0;k<4;++k){
        int xn=bx+dx4[k], yn=by+dy4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        float w = contrast[(size_t)idxAt(xn,yn,tilesX)] - contrast[(size_t)g.idx];
        float nx,ny; std::tie(nx,ny)=ndcOf(xn,yn); float vx=nx-cxN, vy=ny-cyN; if(norm2(vx,vy)){ gx+=w*vx; gy+=w*vy; }
    }
    float dirx=(std::fabs(gx)+std::fabs(gy)>0)?gx:g.ndcX, diry=(std::fabs(gx)+std::fabs(gy)>0)?gy:g.ndcY;
    norm2(dirx,diry);

    // Drift-Momentum (stabilisiert gegen Flip-Flop)
    if(g.vx||g.vy){ dirx = kDIR_MOMENTUM*g.vx + (1.f-kDIR_MOMENTUM)*dirx; diry = kDIR_MOMENTUM*g.vy + (1.f-kDIR_MOMENTUM)*diry; norm2(dirx,diry); }

    // Turn-Limiter
    if(g.vx||g.vy){
        float turn=angleDeg(g.vx,g.vy,dirx,diry);
        if(turn>kMAX_TURN_DEG){
            const float t = kMAX_TURN_DEG/turn;
            dirx = (1.f-t)*g.vx + t*dirx; diry=(1.f-t)*g.vy + t*diry; norm2(dirx,diry);
        }
        if(turn>kHARD_FLIP_DEG){ // starker Flip ⇒ Schritt stark dämpfen
            g.vx *= 0.5f; g.vy *= 0.5f;
        }
    }
    g.vx = 0.7f*g.vx + 0.3f*dirx; g.vy = 0.7f*g.vy + 0.3f*diry;

    // grobe NDC-Schrittweite
    const float stepNdc = clampf(kSEED_STEP_NDC*(0.8f+0.4f*std::sqrt(g.ndcX*g.ndcX+g.ndcY*g.ndcY)), 0.5f*kSEED_STEP_NDC, kSTEP_MAX_NDC);
    const float invZ = 1.f/std::max(1e-6f, zoom);

    out.shouldZoom=true;
    out.bestIndex=g.idx;
    out.isNewTarget=true;
    out.newOffsetX = prevOff.x + dirx*(stepNdc*invZ);
    out.newOffsetY = prevOff.y + diry*(stepNdc*invZ);
    out.distance   = stepNdc*invZ;
    return out;
}

void evaluateAndApply(::FrameContext& fctx, ::RendererState& state, ZoomState& /*bus*/, float /*gain*/) noexcept
{
    beginFrame();
    if(CudaInterop::getPauseZoom()){ fctx.shouldZoom=false; fctx.newOffset=fctx.offset; fctx.newOffsetD={ (double)fctx.offset.x,(double)fctx.offset.y }; return; }

    // Overlay-Grid
    const int tilePx = std::max(1, (Settings::Kolibri::gridScreenConstant? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + tilePx - 1)/tilePx;
    const int tilesY = (fctx.height + tilePx - 1)/tilePx;

    const float2 prevF = fctx.offset;
    ZoomResult zr = evaluateZoomTarget(state.h_entropy, state.h_contrast, tilesX, tilesY,
                                       fctx.width, fctx.height, fctx.offset, fctx.zoom, prevF, *(ZoomState*)nullptr);

    if(!zr.shouldZoom){
        fctx.shouldZoom=false; fctx.newOffset=prevF; fctx.newOffsetD={ (double)prevF.x,(double)prevF.y }; return;
    }

    // Pixel->Welt Schritt
    double stepX=0, stepY=0;
    capy_pixel_steps_from_zoom_scale((double)state.pixelScale.x,(double)state.pixelScale.y,
                                     fctx.width, fctx.zoomD>0.0?fctx.zoomD:(double)state.zoom, stepX, stepY);

    // Richtung aus zr (normiert)
    double dx=(double)(zr.newOffsetX - prevF.x), dy=(double)(zr.newOffsetY - prevF.y);
    { const double n2=dx*dx+dy*dy; if(!(n2>0.0)){ dx=1.0; dy=0.0; } else { const double inv=1.0/std::sqrt(n2); dx*=inv; dy*=inv; } }

    const double halfW = 0.5*(double)fctx.width  * std::fabs(stepX);
    const double halfH = 0.5*(double)fctx.height * std::fabs(stepY);

    // kleine, feste Weltbewegung aus NDC
    const double stepNdc = (double)clampf(kSEED_STEP_NDC, 0.5f*kSEED_STEP_NDC, kSTEP_MAX_NDC);
    double mvX = dx*(stepNdc*halfW), mvY = dy*(stepNdc*halfH);

    // min. 0.5 Pixel
    const double minPix = std::min(std::fabs(stepX), std::fabs(stepY));
    const double len = std::hypot(mvX,mvY);
    if(len < 0.5*minPix && len>0.0){ double s=(0.5*minPix)/len; mvX*=s; mvY*=s; }

    // In-Set-Veto (Zielzentrum der Lock-Kachel)
    if(zr.bestIndex>=0){
        float ndcX=0, ndcY=0; tileIndexToNdcCenter(tilesX,tilesY,zr.bestIndex,ndcX,ndcY);
        const double px = (double)ndcX*0.5*(double)fctx.width, py=(double)ndcY*0.5*(double)fctx.height;
        const double wx = (double)state.center.x + px*stepX, wy=(double)state.center.y + py*stepY;
        if(insideCardioidOrBulb(wx,wy)){ mvX*=0.6; mvY*=0.6; } // sanft abbremsen statt springen
    }

    // leichte EMA aufs Center
    const double baseX=(double)state.center.x, baseY=(double)state.center.y;
    const double tx = baseX*(1.0-(double)kBLEND_A) + (baseX+mvX)*(double)kBLEND_A;
    const double ty = baseY*(1.0-(double)kBLEND_A) + (baseY+mvY)*(double)kBLEND_A;

    fctx.shouldZoom=true;
    fctx.newOffsetD={tx,ty};
    fctx.newOffset = make_float2((float)tx,(float)ty);
}

} // namespace ZoomLogic
