///// Otter: Silk-Lite auto-pan/zoom controller (lokal, kantengetrieben, ohne Heatmap-Zwang).
///// Schneefuchs: Sanfte Richtungswechsel via Lock+Cooldown+Hysterese + leichter EMA; keine API-Drifts.
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

constexpr float  kSEED_STEP_NDC = 0.012f, kSTEP_MAX_NDC = 0.35f, kBLEND_A = 0.16f;
constexpr int    kSEARCH_R = 2, kCOOLDOWN_FRAMES = 24, kREQUIRED_FRAMES = 3;
constexpr float  kTHR_MED_ADD_MAD = 0.90f, kHYST = 1.60f, kMUCH = 1.90f;
constexpr float  kMIN_RETARGET = 0.10f, kLOCKBOX = 0.18f, kTURN_MAX = 15.0f, kTURN_STRONG = 90.0f;
constexpr double kMAX_PX_MOVE = 6.0;

inline float clampf(float x,float a,float b){return x<a?a:(x>b?b:x);}
inline bool norm2(float& x,float& y){float n=x*x+y*y; if(!(n>0.f)) return false; float inv=1.f/std::sqrt(n); x*=inv; y*=inv; return true;}

inline bool insideCardioidOrBulb(double x,double y) noexcept {
    const double xm=x-0.25,q=xm*xm+y*y; if(q*(q+xm)<0.25*y*y) return true; const double dx=x+1.0; return (dx*dx+y*y)<0.0625;
}

float median_inplace(std::vector<float>& v){
    if(v.empty()) return 0.f; size_t m=v.size()/2;
    std::nth_element(v.begin(),v.begin()+m,v.end()); float med=v[m];
    if(!(v.size()&1)){ std::nth_element(v.begin(),v.begin()+m-1,v.begin()+m); med=0.5f*(med+v[m-1]); }
    return med;
}
float mad_from_center_inplace(std::vector<float>& v,float c){ for(auto& x: v) x=std::fabs(x-c); return std::max(1e-6f, median_inplace(v)); }

inline void tileIndexToNdcCenter(int tx,int ty,int idx,float& nx,float& ny){
    int x = (tx>0)? idx%tx : 0, y = (tx>0)? idx/tx : 0;
    float cx = (x+0.5f)/std::max(1,tx), cy = (y+0.5f)/std::max(1,ty);
    nx = cx*2.f-1.f; ny = cy*2.f-1.f;
}
inline int idxAt(int x,int y,int tx){ return y*tx + x; }

inline float edgeScoreAt(const std::vector<float>& c,int x,int y,int tx,int ty){
    int i=idxAt(x,y,tx); float lc=c[(size_t)i], sum=0.f; int n=0;
    int nx4[4]={x-1,x+1,x,x}, ny4[4]={y,y,y-1,y+1};
    for(int k=0;k<4;++k){ int xn=nx4[k],yn=ny4[k]; if(xn<0||yn<0||xn>=tx||yn>=ty) continue; sum += std::fabs(c[(size_t)idxAt(xn,yn,tx)]-lc); ++n; }
    float nd=(n>0)?(sum/n):0.f; return lc + 0.75f*nd;
}

struct ZLock{
    bool init=false; int lockIdx=-1,candIdx=-1,candFrames=0,cooldown=0;
    float lockScore=0.f, lockX=0.f, lockY=0.f, candScore=0.f, candX=0.f, candY=0.f, vx=0.f, vy=0.f, gx=0.f, gy=0.f;
};
static ZLock g;

inline void beginFrame(){ if(g.cooldown>0) g.cooldown--; g.vx*=0.9f; g.vy*=0.9f; }
inline float angleDeg(float ax,float ay,float bx,float by){
    float a2=ax*ax+ay*ay,b2=bx*bx+by*by; if(!(a2>0.f)||!(b2>0.f)) return 0.f;
    float c=(ax*bx+ay*by)/(std::sqrt(a2)*std::sqrt(b2)); c = c>1.f?1.f:(c<-1.f?-1.f:c);
    return std::acos(c)*180.f/3.14159265358979323846f;
}
inline void blendDir(float px,float py,float dx,float dy,float t,float& ox,float& oy){
    norm2(px,py); norm2(dx,dy); ox=(1.f-t)*px + t*dx; oy=(1.f-t)*py + t*dy; norm2(ox,oy);
}

} // namespace

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>&/*entropy*/,
                              const std::vector<float>& contrast,
                              int tilesX,int tilesY,
                              int/*w*/,int/*h*/,
                              float2/*curOffset*/,float zoom,
                              float2 prev,
                              ZoomState&/*state*/) noexcept
{
    ZoomResult out{};
    int total = (tilesX>0 && tilesY>0)? tilesX*tilesY : 0;
    if(total<=0 || (int)contrast.size()<total){ out.shouldZoom=Settings::ForceAlwaysZoom; return out; }

    std::vector<float> c = contrast; float med=median_inplace(c); float mad=mad_from_center_inplace(c,med);
    float thr = med + kTHR_MED_ADD_MAD*mad;

    int cx=tilesX/2, cy=tilesY/2, bestI=-1; float bestS=-1e30f,bestD2=1e30f,bxN=0.f,byN=0.f;
    for(int dy=-kSEARCH_R; dy<=kSEARCH_R; ++dy) for(int dx=-kSEARCH_R; dx<=kSEARCH_R; ++dx){
        int x=cx+dx,y=cy+dy; if(x<0||y<0||x>=tilesX||y>=tilesY) continue;
        float s=edgeScoreAt(contrast,x,y,tilesX,tilesY); if(!(s>thr)) continue;
        float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(x,y,tilesX),nx,ny); float d2=nx*nx+ny*ny;
        if(d2<bestD2-1e-6f || (std::fabs(d2-bestD2)<=1e-6f && s>bestS)){ bestD2=d2; bestS=s; bestI=idxAt(x,y,tilesX); bxN=nx; byN=ny; }
    }

    if(!g.init && bestI>=0){ g={true,bestI,-1,0,kCOOLDOWN_FRAMES,bestS,bxN,byN,0,0,0,0,0,0,0}; }

    if(bestI<0){
        out.shouldZoom=Settings::ForceAlwaysZoom;
        float dx= g.init? g.lockX : 1.f, dy= g.init? g.lockY : 0.f; norm2(dx,dy);
        float invZ = 1.f/std::max(1e-6f,zoom), step=clampf(kSEED_STEP_NDC,0.f,kSTEP_MAX_NDC);
        out.newOffsetX = prev.x + dx*(step*invZ); out.newOffsetY = prev.y + dy*(step*invZ); out.distance=step*invZ; return out;
    }

    if(bestI==g.candIdx){ g.candFrames++; if(bestS>g.candScore){ g.candScore=bestS; g.candX=bxN; g.candY=byN; } }
    else{ g.candIdx=bestI; g.candScore=bestS; g.candX=bxN; g.candY=byN; g.candFrames=1; }

    float dX=g.candX-g.lockX, dY=g.candY-g.lockY, d=std::sqrt(dX*dX+dY*dY);
    bool far = d>=kMIN_RETARGET, better = g.candScore>=g.lockScore*kHYST, much=g.candScore>=g.lockScore*kMUCH, inBox = d<=kLOCKBOX;
    if(g.init){
        bool hold = g.candFrames>=kREQUIRED_FRAMES;
        if( ((g.cooldown<=0)&&far&&better&&hold) || much ){
            if(!inBox || much){
                if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[ZOOM][RETARGET] %d->%d d=%.3f s=%.3f->%.3f fr=%d",
                    g.lockIdx,g.candIdx,(double)d,(double)g.lockScore,(double)g.candScore,g.candFrames);
                g.lockIdx=g.candIdx; g.lockScore=0.7f*g.lockScore+0.3f*g.candScore; g.lockX=g.candX; g.lockY=g.candY; g.cooldown=kCOOLDOWN_FRAMES;
                g.candIdx=-1; g.candFrames=0;
            }
        }
    }

    int lx=g.lockIdx%tilesX, ly=g.lockIdx/tilesX;
    auto ndcOf=[&](int X,int Y){ float nx,ny; tileIndexToNdcCenter(tilesX,tilesY,idxAt(X,Y,tilesX),nx,ny); return make_float2(nx,ny); };
    float2 pC=ndcOf(lx,ly), grad=make_float2(0,0); int nx4[4]={lx-1,lx+1,lx,lx}, ny4[4]={ly,ly,ly-1,ly+1};
    for(int k=0;k<4;++k){ int xn=nx4[k],yn=ny4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        float w=contrast[(size_t)idxAt(xn,yn,tilesX)]-contrast[(size_t)g.lockIdx]; float2 pN=ndcOf(xn,yn);
        float vx=pN.x-pC.x, vy=pN.y-pC.y; if(norm2(vx,vy)){ grad.x+=w*vx; grad.y+=w*vy; } }

    float dirx,diry;
    if(std::fabs(grad.x)+std::fabs(grad.y)>0.f){ dirx=grad.x; diry=grad.y; norm2(dirx,diry); }
    else { dirx=g.lockX; diry=g.lockY; if(!norm2(dirx,diry)){dirx=1.f; diry=0.f;} }

    if(std::fabs(g.vx)+std::fabs(g.vy)>0.f){
        float turn=angleDeg(g.vx,g.vy,dirx,diry);
        if(turn>kTURN_MAX){ float ox,oy; blendDir(g.vx,g.vy,dirx,diry,kTURN_MAX/turn,ox,oy); dirx=ox; diry=oy; }
        if(turn>kTURN_STRONG){ g.vx*=0.3f; g.vy*=0.3f; }
    }
    g.vx = 0.7f*g.vx + 0.3f*dirx; g.vy = 0.7f*g.vy + 0.3f*diry;

    float invZ=1.f/std::max(1e-6f,zoom), step=clampf(kSEED_STEP_NDC,0.f,kSTEP_MAX_NDC);
    out.shouldZoom=true; out.bestIndex=g.lockIdx; out.isNewTarget=true;
    out.newOffsetX = prev.x + dirx*(step*invZ);
    out.newOffsetY = prev.y + diry*(step*invZ);
    out.distance   = step*invZ;
    return out;
}

void evaluateAndApply(::FrameContext& fctx, ::RendererState& state, ZoomState&/*bus*/, float/*gain*/) noexcept
{
    beginFrame();
    if(CudaInterop::getPauseZoom()){ fctx.shouldZoom=false; fctx.newOffset=fctx.offset; fctx.newOffsetD={(double)fctx.offset.x,(double)fctx.offset.y}; return; }

    const int overlayTilePx = std::max(1, (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
    const int tilesY = (fctx.height + overlayTilePx - 1) / overlayTilePx;

    const float2 prevF = fctx.offset;
    ZoomResult zr = evaluateZoomTarget(state.h_entropy,state.h_contrast,tilesX,tilesY,fctx.width,fctx.height,fctx.offset,fctx.zoom,prevF,*(ZoomState*)nullptr);

    if(!zr.shouldZoom){ fctx.shouldZoom=false; fctx.newOffset=prevF; fctx.newOffsetD={(double)prevF.x,(double)prevF.y}; return; }

    double stepX=0.0, stepY=0.0;
    capy_pixel_steps_from_zoom_scale((double)state.pixelScale.x,(double)state.pixelScale.y,fctx.width,
                                     fctx.zoomD>0.0?fctx.zoomD:(double)state.zoom, stepX,stepY);

    double dirx = (double)(zr.newOffsetX - prevF.x), diry = (double)(zr.newOffsetY - prevF.y);
    { double n2=dirx*dirx+diry*diry; if(!(n2>0.0)){ dirx=1.0; diry=0.0; } else { double inv=1.0/std::sqrt(n2); dirx*=inv; diry*=inv; } }

    double halfW = 0.5 * (double)fctx.width  * std::fabs(stepX);
    double halfH = 0.5 * (double)fctx.height * std::fabs(stepY);

    float ndcTX=0.f, ndcTY=0.f; if(zr.bestIndex>=0) tileIndexToNdcCenter(tilesX,tilesY,zr.bestIndex,ndcTX,ndcTY);
    float distNdc = std::sqrt(ndcTX*ndcTX+ndcTY*ndcTY);
    double stepNdc = (double)clampf(kSEED_STEP_NDC*(0.6f+0.8f*distNdc),0.6f*kSEED_STEP_NDC,kSTEP_MAX_NDC);

    double moveX = dirx*(stepNdc*halfW), moveY = diry*(stepNdc*halfH);

    const double minPixStep = std::min(std::fabs(stepX),std::fabs(stepY));
    double movLen = std::hypot(moveX,moveY);
    if(movLen < 0.5*minPixStep && movLen>0.0){ double s=(0.5*minPixStep)/movLen; moveX*=s; moveY*=s; }
    else if(!(movLen>0.0)){ moveX=minPixStep; moveY=0.0; }

    // Pixel-Move-Cap
    { double capX=kMAX_PX_MOVE*std::fabs(stepX), capY=kMAX_PX_MOVE*std::fabs(stepY);
      if(std::fabs(moveX)>capX) moveX=(moveX>0?capX:-capX);
      if(std::fabs(moveY)>capY) moveY=(moveY>0?capY:-capY); }

    // In-Set-Veto (kompakt)
    if(zr.bestIndex>=0){
        int bx=zr.bestIndex%tilesX, by=zr.bestIndex/tilesX;
        double px=(double)ndcTX*0.5*(double)fctx.width, py=(double)ndcTY*0.5*(double)fctx.height;
        double wx=(double)state.center.x + px*stepX, wy=(double)state.center.y + py*stepY;
        if(insideCardioidOrBulb(wx,wy)){
            int bestN=-1; float bestS=-1e30f, nx=0,ny=0; int nx4[4]={bx-1,bx+1,bx,bx}, ny4[4]={by,by,by-1,by+1};
            for(int k=0;k<4;++k){ int xn=nx4[k],yn=ny4[k]; if(xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
                float s=edgeScoreAt(state.h_contrast,xn,yn,tilesX,tilesY); if(s>bestS){bestS=s; bestN=idxAt(xn,yn,tilesX); tileIndexToNdcCenter(tilesX,tilesY,bestN,nx,ny);} }
            if(bestN>=0){ double dx=nx,dy=ny,n2=dx*dx+dy*dy; if(n2>0.0){ double inv=1.0/std::sqrt(n2); dx*=inv; dy*=inv; }
                moveX=dx*(stepNdc*halfW); moveY=dy*(stepNdc*halfH); }
        }
    }

    double baseX=(double)state.center.x, baseY=(double)state.center.y;
    double txD = baseX*(1.0-kBLEND_A) + (baseX+moveX)*kBLEND_A;
    double tyD = baseY*(1.0-kBLEND_A) + (baseY+moveY)*kBLEND_A;

    fctx.shouldZoom=true; fctx.newOffsetD={txD,tyD}; fctx.newOffset=make_float2((float)txD,(float)tyD);
}

} // namespace ZoomLogic
