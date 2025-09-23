///// Otter: Silk-Lite zoom controller with PD motion and hysteresis; ASCII-only logs.
///// Schneefuchs: No API drift; statistics cadence and retarget intervals documented.
///// Maus: ForceAlwaysZoom default ON; warm-up freeze respected.
///// Datei: src/zoom_logic.cpp

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp" // tileIndexToPixelCenter
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

// Adapter-Dependencies (Renderer/Interop) — benötigt für evaluateAndApply(...)
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "cuda_interop.hpp"

namespace {
constexpr float kALPHA_E=1.0f, kBETA_C=0.5f, kTEMP_BASE=1.0f, kMIN_SIGNAL_STD=0.15f;
constexpr float kSTD_HI=0.18f, kSTD_LO=0.12f;
constexpr int   kM0_ITER=96; constexpr float kMETRIC_ALPHA=0.40f, kMETRIC_MIN_F=0.55f, kMETRIC_MAX_F=1.00f;
constexpr float kWARMUP_DRIFT_NDC=0.08f, kSEED_STEP_NDC=0.015f;
constexpr float kTURN_OMEGA_MIN=2.5f, kTURN_OMEGA_MAX=7.0f, kTHETA_DAMP_LO=0.42f, kTHETA_DAMP_HI=1.32f;
constexpr float kSOFTMAX_LOG_EPS=-7.0f; constexpr int kMIN_TOPK=7;
constexpr float kEMA_TAU_MIN=0.040f, kEMA_TAU_MAX=0.220f, kEMA_ALPHA_MIN=0.06f, kEMA_ALPHA_MAX=0.30f, kFORCE_MIN_DRIFT_ALPHA=0.06f;
constexpr float kSTEP_MAX_NDC=0.35f, kHANDOVER_SECONDS=0.30f;

inline float clampf(float x,float a,float b){return x<a?a:(x>b?b:x);}
inline float smoothstepf(float a,float b,float x){const float t=clampf((x-a)/(b-a),0.0f,1.0f);return t*t*(3.0f-2.0f*t);}
inline bool normalize2D(float& x,float& y){const float n2=x*x+y*y; if(n2<=1e-20f) return false; const float inv=1.0f/std::sqrt(n2); x*=inv; y*=inv; return true;}
inline void rotateTowardsLimited(float& dx,float& dy,float tx,float ty,float maxA){
    if(!normalize2D(tx,ty)) return; if(!normalize2D(dx,dy)){dx=tx;dy=ty;return;}
    const float dot=clampf(dx*tx+dy*ty,-1.0f,1.0f); const float ang=std::sqrt(std::max(0.0f,2.0f*(1.0f-dot)));
    if(!(ang>0.0f)||ang<=maxA){dx=tx;dy=ty;return;}
    const float t=clampf(maxA/ang,0.0f,1.0f); float nx=(1.0f-t)*dx+t*tx, ny=(1.0f-t)*dy+t*ty; if(!normalize2D(nx,ny)){nx=tx;ny=ty;} dx=nx; dy=ny;
}
inline bool insideCardioidOrBulb(double x,double y) noexcept {
    const double xm=x-0.25, q=xm*xm+y*y; if(q*(q+xm)<0.25*y*y) return true; const double dx=x+1.0; return dx*dx+y*y<0.0625;
}
inline void antiVoidDriftNDC(float cx,float cy,float& nx,float& ny){float vx=0.5f*(cx-0.25f)+0.5f*(cx+1.0f), vy=cy; if(!normalize2D(vx,vy)){vx=1.0f;vy=0.0f;} nx=vx; ny=vy;}
float median_inplace(std::vector<float>& v){ if(v.empty()) return 0.0f; size_t n=v.size(), m=n/2; std::nth_element(v.begin(),v.begin()+m,v.end()); float med=v[m]; if((n&1)==0){ std::nth_element(v.begin(),v.begin()+m-1,v.begin()+m); med=0.5f*(med+v[m-1]); } return med; }
float mad_from_center_inplace(std::vector<float>& v,float med){ if(v.empty()) return 1.0f; for(float& x:v) x=std::fabs(x-med); float mad=median_inplace(v); return mad>1e-6f?mad:1.0f; }
double centerDzdcMag(double cx,double cy,int maxIter=kM0_ITER){
    double zx=0,zy=0,dx=0,dy=0,cx0=cx,cy0=cy; for(int i=0;i<maxIter;++i){ double tdx=2.0*(zx*dx-zy*dy)+1.0, tdy=2.0*(zx*dy+zy*dx); dx=tdx; dy=tdy; double nzx=zx*zx-zy*zy+cx0, nzy=2.0*zx*zy+cy0; zx=nzx; zy=nzy; if(zx*zx+zy*zy>4.0) break; } return std::sqrt(dx*dx+dy*dy);
}
thread_local bool  g_dirInit=false, g_signalLast=true, g_inHandover=false;
thread_local float g_prevDirX=1.0f, g_prevDirY=0.0f, g_prevOmega=kTURN_OMEGA_MIN, g_tempEma=kTEMP_BASE, g_handoverT=0.0f;
} // namespace

namespace ZoomLogic {

float computeEntropyContrast(const std::vector<float>& entropy,int w,int h,int ts) noexcept {
    if(w<=0||h<=0||ts<=0) return 0.0f; const int txs=(w+ts-1)/ts, tys=(h+ts-1)/ts, tot=txs*tys; if(tot<=0||(int)entropy.size()<tot) return 0.0f;
    double acc=0; int cnt=0;
    for(int ty=0;ty<tys;++ty){
        for(int tx=0;tx<txs;++tx){
            const int i=ty*txs+tx; const float centerE=entropy[i]; float sum=0; int n=0;
            const int nx[4]={tx-1,tx+1,tx,tx}, ny[4]={ty,ty,ty-1,ty+1};
            for(int k=0;k<4;++k){ if(nx[k]<0||ny[k]<0||nx[k]>=txs||ny[k]>=tys) continue; sum+=std::fabs(entropy[ny[k]*txs+nx[k]]-centerE); ++n; }
            if(n>0){ acc+=sum/n; ++cnt; }
        }
    }
    return cnt>0? (float)(acc/cnt) : 0.0f;
}

ZoomResult evaluateZoomTarget(const std::vector<float>& entropy,const std::vector<float>& contrast,
    int tilesX,int tilesY,int width,int height, float2 currentOffset,float zoom, float2 previousOffset, ZoomState& state) noexcept
{
    using clock=std::chrono::steady_clock; const auto t0=clock::now();
    static clock::time_point s_last; static bool s_have=false;
    double dt = s_have ? std::chrono::duration<double>(t0-s_last).count() : (1.0/60.0); s_last=t0; s_have=true; if(!(dt>0.0)) dt=1.0/60.0;
    // Spike-Schutz: für Dynamik (Omega/EMA) geklammertes dt verwenden; Handover nutzt reales dt
    const double dt_clamped = clampf((float)dt, 1.0f/240.0f, 1.0f/24.0f);

    // Warm-up / handover
    static bool warmInit=false; static clock::time_point warmStart; if(!warmInit){warmStart=t0; warmInit=true;}
    const double warmTime=std::chrono::duration<double>(t0-warmStart).count(); const bool inFreeze=(warmTime<Settings::warmUpFreezeSeconds);
    if(g_inHandover && inFreeze) g_handoverT=0.0f; else if(!inFreeze && !g_inHandover){ g_inHandover=true; g_handoverT=0.0f; }

    ZoomResult out{}; out.bestIndex=-1; out.shouldZoom=false; out.isNewTarget=false; out.newOffset=previousOffset; out.minDistance=0.02f;

    const int total=tilesX*tilesY; if(tilesX<=0||tilesY<=0||total<=0||(int)entropy.size()<total||(int)contrast.size()<total){ out.shouldZoom=Settings::ForceAlwaysZoom; return out; }

    const double invW=width>0? 1.0/double(width):0.0, invH=height>0? 1.0/double(height):0.0;
    const double invZ = 1.0/std::max(1e-6f,zoom);

    // Metric factor: spare centerDzdcMag wenn klar "innen"
    const bool centerInside = insideCardioidOrBulb(currentOffset.x,currentOffset.y);
    double mFac = kMETRIC_MAX_F;
    if(!centerInside){
        const double M0  = centerDzdcMag((double)currentOffset.x,(double)currentOffset.y,kM0_ITER);
        const double M0c = std::log1p(std::max(0.0, M0));
        mFac = std::clamp(1.0/(1.0+(double)kMETRIC_ALPHA*M0c),(double)kMETRIC_MIN_F,(double)kMETRIC_MAX_F);
    }
    const double invZE = invZ*mFac;

    if(inFreeze){
        out.shouldZoom=true;
        if(insideCardioidOrBulb(currentOffset.x,currentOffset.y)){
            float nx=1.0f,ny=0.0f; antiVoidDriftNDC(currentOffset.x,currentOffset.y,nx,ny);
            const float2 t=make_float2(previousOffset.x+nx*(float)(kWARMUP_DRIFT_NDC*invZE), previousOffset.y+ny*(float)(kWARMUP_DRIFT_NDC*invZE));
            const float a=0.20f; out.newOffset=make_float2(previousOffset.x*(1-a)+t.x*a, previousOffset.y*(1-a)+t.y*a);
        }else{
            float sx=g_dirInit?g_prevDirX:1.0f, sy=g_dirInit?g_prevDirY:0.0f;
            if(!normalize2D(sx,sy)){ sx=1.0f; sy=0.0f; }
            const float2 t=make_float2(previousOffset.x+sx*(float)(kSEED_STEP_NDC*invZE), previousOffset.y+sy*(float)(kSEED_STEP_NDC*invZE));
            const float a=0.20f; out.newOffset=make_float2(previousOffset.x*(1-a)+t.x*a, previousOffset.y*(1-a)+t.y*a);
        }
        const float dx=out.newOffset.x-previousOffset.x, dy=out.newOffset.y-previousOffset.y; out.distance=std::sqrt(dx*dx+dy*dy); return out;
    }

    // Median/MAD (alloc-light)
    thread_local std::vector<float> e, contr; e.clear(); contr.clear(); e.reserve((size_t)total); contr.reserve((size_t)total);
    e.insert(e.end(),entropy.begin(),entropy.begin()+total);
    contr.insert(contr.end(),contrast.begin(),contrast.begin()+total);
    const float eMed=median_inplace(e), eMad=mad_from_center_inplace(e,eMed);
    const float cMed=median_inplace(contr), cMad=mad_from_center_inplace(contr,cMed);

    // One-pass stats + top-k floor
    float best=-1e9f; int bestIdx=-1; double m=0, m2=0;
    float topK[kMIN_TOPK]; for(int i=0;i<kMIN_TOPK;++i) topK[i]=-1e9f;
    auto pushTop=[&](float s){ int p=0; for(int i=1;i<kMIN_TOPK;++i) if(topK[i]<topK[p]) p=i; if(s>topK[p]) topK[p]=s; };

    for(int i=0;i<total;++i){ const float ez=(entropy[i]-eMed)/eMad, cz=(contrast[i]-cMed)/cMad, s=kALPHA_E*ez+kBETA_C*cz; m+=s; m2+=double(s)*s; if(s>best){best=s;bestIdx=i;} pushTop(s); }
    m/=std::max(1,total); const double var=std::max(0.0, m2/std::max(1,total) - m*m), stdS=std::sqrt(var);

    // temp with EMA
    float tempRaw=kTEMP_BASE; if(stdS>1e-6) tempRaw=(float)(kTEMP_BASE/(0.5f+(float)stdS)); tempRaw=clampf(tempRaw,0.2f,2.5f);
    const float temp=g_tempEma=0.8f*g_tempEma+0.2f*tempRaw;

    const float sCut=best+temp*kSOFTMAX_LOG_EPS; float topMin=topK[0]; for(int i=1;i<kMIN_TOPK;++i) topMin=std::min(topMin,topK[i]);
    const float sFloor = (total>=kMIN_TOPK)? std::min(sCut,topMin) : sCut;
    const float invT = 1.0f/std::max(1e-6f,temp);

    // Softmax accumulation
    double sumW=0,numX=0,numY=0; int bestAdj=-1; float bestAdjS=-1e9f;
    for(int i=0;i<total;++i){
        const float ez=(entropy[i]-eMed)/eMad, cz=(contrast[i]-cMed)/cMad, s=kALPHA_E*ez+kBETA_C*cz; if(s<sFloor) continue;
        auto p=tileIndexToPixelCenter(i,tilesX,tilesY,width,height);
        const double ndcX=(double(p.first)*invW-0.5)*2.0, ndcY=(double(p.second)*invH-0.5)*2.0;
        const double tx=currentOffset.x+ndcX*invZE, ty=currentOffset.y+ndcY*invZE; if(insideCardioidOrBulb(tx,ty)) continue;
        const double w=std::exp(double((s-best)*invT)); sumW+=w; numX+=w*ndcX; numY+=w*ndcY; if(s>bestAdjS){bestAdjS=s;bestAdj=i;}
    }

    // Choose NDC target
    double ndcTX=0.0, ndcTY=0.0;
    if(sumW>0.0){ const double inv=1.0/sumW; ndcTX=numX*inv; ndcTY=numY*inv; }
    else if(bestAdj>=0){ auto p=tileIndexToPixelCenter(bestAdj,tilesX,tilesY,width,height); ndcTX=(double(p.first)*invW-0.5)*2.0; ndcTY=(double(p.second)*invH-0.5)*2.0; }
    else if(bestIdx>=0){ auto p=tileIndexToPixelCenter(bestIdx,tilesX,tilesY,width,height); double bx=(double(p.first)*invW-0.5)*2.0, by=(double(p.second)*invH-0.5)*2.0; double tx=currentOffset.x+bx*invZE, ty=currentOffset.y+by*invZE; if(!insideCardioidOrBulb(tx,ty)){ ndcTX=bx; ndcTY=by; } }
    // robustes Nullziel (Epsilon)
    if (std::fabs(ndcTX) + std::fabs(ndcTY) < 1e-9) {
        float fx = g_dirInit ? g_prevDirX : 1.0f;
        float fy = g_dirInit ? g_prevDirY : 0.0f;
        if(!normalize2D(fx,fy)){ fx=1.0f; fy=0.0f; }
        ndcTX = fx; ndcTY = fy;
    }

    // Patch-A: angle-aware inertia + deadband
    if(g_dirInit){
        float nx=(float)ndcTX, ny=(float)ndcTY;
        if(normalize2D(nx,ny)){
            const float cosA=clampf(nx*g_prevDirX+ny*g_prevDirY,-1.0f,1.0f);
            if(cosA>0.9986f){ ndcTX=g_prevDirX; ndcTY=g_prevDirY; } // ~3°
            else{ const float t=clampf((cosA-0.9063f)/(0.9962f-0.9063f),0.0f,1.0f), wOld=0.15f+0.35f*t; ndcTX=(1.0f-wOld)*ndcTX+wOld*g_prevDirX; ndcTY=(1.0f-wOld)*ndcTY+wOld*g_prevDirY; }
        }
    }

    // Build proposed
    const float2 rawT=make_float2(previousOffset.x+(float)(ndcTX*invZE), previousOffset.y+(float)(ndcTY*invZE));
    float mvx=rawT.x-previousOffset.x, mvy=rawT.y-previousOffset.y; const float r2=mvx*mvx+mvy*mvy, r=(r2>0.0f)?std::sqrt(r2):0.0f;
    float dirX=g_dirInit?g_prevDirX:(r>0.0f?mvx/r:1.0f), dirY=g_dirInit?g_prevDirY:(r>0.0f?mvy/r:0.0f); g_dirInit=true;
    float tgtX=mvx, tgtY=mvy; const bool hasMove=normalize2D(tgtX,tgtY);

    const float sigF=clampf((float)stdS,0.0f,1.0f), distF=clampf(r/0.25f,0.0f,1.0f);
    const float omegaRaw=kTURN_OMEGA_MIN+(kTURN_OMEGA_MAX-kTURN_OMEGA_MIN)*std::max(sigF,distF);
    const float omega=g_prevOmega=0.7f*g_prevOmega+0.3f*omegaRaw; const float maxTurn=omega*(float)dt_clamped;

    float lenScale=1.0f;
    if(hasMove){
        const float d=clampf(dirX*tgtX+dirY*tgtY,-1.0f,1.0f);
        const float ang=std::sqrt(std::max(0.0f, 2.0f*(1.0f - d))); // Korrektur: 2*(1 - cos)
        rotateTowardsLimited(dirX,dirY,tgtX,tgtY,maxTurn);
        lenScale=1.0f-smoothstepf(kTHETA_DAMP_LO,kTHETA_DAMP_HI,ang);
        g_prevDirX=dirX; g_prevDirY=dirY;
    }

    float2 proposed=make_float2(previousOffset.x+dirX*(r*lenScale), previousOffset.y+dirY*(r*lenScale));

    // Handover blend (~300 ms) nach Freeze – Zeitfortschritt mit realem dt
    if(g_inHandover){
        g_handoverT+=(float)dt; const float t=clampf(g_handoverT/kHANDOVER_SECONDS,0.0f,1.0f);
        float sx=g_prevDirX, sy=g_prevDirY; if(!normalize2D(sx,sy)){sx=1.0f;sy=0.0f;}
        const float2 seed=make_float2(previousOffset.x+sx*(float)(0.5f*kSEED_STEP_NDC*invZE), previousOffset.y+sy*(float)(0.5f*kSEED_STEP_NDC*invZE));
        proposed=make_float2(seed.x*(1.0f - t) + proposed.x*t, seed.y*(1.0f - t) + proposed.y*t);
        if(t>=1.0f) g_inHandover=false;
    }

    // EMA (distance-adaptive) – mit dt_clamped
    const float dxP=proposed.x-previousOffset.x, dyP=proposed.y-previousOffset.y, d=std::sqrt(dxP*dxP+dyP*dyP);
    const float dn=clampf(d/0.5f,0.0f,1.0f), tau=kEMA_TAU_MAX+(kEMA_TAU_MIN-kEMA_TAU_MAX)*dn;
    float a=1.0f-std::exp(-(float)dt_clamped/std::max(1e-5f,tau)); a=clampf(a,kEMA_ALPHA_MIN,kEMA_ALPHA_MAX);
    if(Settings::ForceAlwaysZoom && stdS<kMIN_SIGNAL_STD) a=std::max(a,kFORCE_MIN_DRIFT_ALPHA);
    float2 smoothed=make_float2(previousOffset.x*(1.0f-a)+proposed.x*a, previousOffset.y*(1.0f-a)+proposed.y*a);

    // Slew limit (cap step/frame) – Unterkante verhindert "Kleben"
    { float ddx=smoothed.x-previousOffset.x, ddy=smoothed.y-previousOffset.y, d2=ddx*ddx+ddy*ddy;
      const float ms_min_abs = 1e-9f;
      const float ms = std::max(kSTEP_MAX_NDC*(float)invZE, ms_min_abs);
      const float ms2=ms*ms; if(d2>ms2 && d2>0.0f){ const float s=ms/std::sqrt(d2); smoothed.x=previousOffset.x+ddx*s; smoothed.y=previousOffset.y+ddy*s; } }

    // Hysteretic signal gate
    const bool gateIn =(stdS>=kSTD_HI), gateOut=(stdS>=kSTD_LO); g_signalLast = g_signalLast ? gateOut : gateIn;
    const bool hasSignal=g_signalLast;

    out.shouldZoom = hasSignal || Settings::ForceAlwaysZoom;
    out.newOffset  = out.shouldZoom ? smoothed : previousOffset;
    { const float ddx=out.newOffset.x-previousOffset.x, ddy=out.newOffset.y-previousOffset.y; out.distance=std::sqrt(ddx*ddx+ddy*ddy); }

    out.bestIndex=-1; out.isNewTarget=false;
    state.lastOffset=out.newOffset; state.lastTilesX=tilesX; state.lastTilesY=tilesY; state.cooldownLeft=0;

    if (Settings::debugLogging){
        const double ndcLen=std::sqrt(ndcTX*ndcTX+ndcTY*ndcTY);
        LUCHS_LOG_HOST("[ZOOM-LITE] invZE=%.3g dist=%.6f ndc=%.3f ema=%.3f omega=%.3f temp=%.3f",
            (float)invZE, out.distance, (float)ndcLen, a, maxTurn, temp);
    }
    return out;
}

// Convenience-Adapter: setzt shouldZoom/newOffset direkt im FrameContext,
// respektiert die globale Pauseflagge aus CudaInterop.
void evaluateAndApply(FrameContext& fctx, RendererState& state, ZoomState& bus, float /*gain*/) noexcept {
    // Pause?
    if (CudaInterop::getPauseZoom()) {
        fctx.shouldZoom = false;
        fctx.newOffset  = fctx.offset;
        return;
    }

    const int tilesX = (fctx.width  + fctx.tileSize - 1) / fctx.tileSize;
    const int tilesY = (fctx.height + fctx.tileSize - 1) / fctx.tileSize;

    const float2 prev = fctx.offset;
    ZoomResult zr = evaluateZoomTarget(
        state.h_entropy, state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,
        prev,
        bus
    );

    fctx.shouldZoom = zr.shouldZoom;
    fctx.newOffset  = zr.newOffset;
}

} // namespace ZoomLogic
