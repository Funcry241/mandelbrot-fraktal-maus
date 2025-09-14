///// Otter: V3-lite OPT — alloc-light (no ndc buffers), single-pass stats for best/mean; identical behavior; no black zoom.
/// /// Schneefuchs: Deterministic, ASCII logs, /WX-safe; same math & constants; micro-opts only (invWidth/Height, fewer divisions).
/// /// Maus: Stable ABI; Patch-A target inertia kept; no planner; HUD markers off.
/// /// Datei: src/zoom_logic.cpp

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp" // tileIndexToPixelCenter
// Nacktmull entfernt: kein settings_nacktmull.hpp mehr

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

namespace {
// ---- Weights & thresholds ----------------------------------------------------
constexpr float kALPHA_E = 1.00f;
constexpr float kBETA_C  = 0.50f;
constexpr float kTEMP_BASE = 1.00f;
constexpr float kMIN_SIGNAL_STD = 0.15f;

// Pullback metric (center)
constexpr int   kM0_ITER       = 96;
constexpr float kMETRIC_ALPHA  = 0.40f;
constexpr float kMETRIC_MIN_F  = 0.55f;
constexpr float kMETRIC_MAX_F  = 1.00f;

// Warm-up / Seed (freeze duration comes from Settings)
constexpr float  kWARMUP_DRIFT_NDC   = 0.08f;
constexpr float  kSEED_STEP_NDC      = 0.015f;

// Turn limiter & length damping
constexpr float kTURN_OMEGA_MIN = 2.5f;
constexpr float kTURN_OMEGA_MAX = 10.0f;
constexpr float kTHETA_DAMP_LO  = 0.42f; // was 0.35f
constexpr float kTHETA_DAMP_HI  = 1.32f; // was 1.20f

constexpr float kSOFTMAX_LOG_EPS = -7.0f;

// EMA (dt)
constexpr float kEMA_TAU_MIN   = 0.040f;
constexpr float kEMA_TAU_MAX   = 0.220f;
constexpr float kEMA_ALPHA_MIN = 0.06f;
constexpr float kEMA_ALPHA_MAX = 0.30f;
constexpr float kFORCE_MIN_DRIFT_ALPHA = 0.06f; // was 0.05f

inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }
inline float smoothstepf(float a,float b,float x){ const float t=clampf((x-a)/(b-a),0.0f,1.0f); return t*t*(3.0f-2.0f*t); }
inline bool normalize2D(float& x,float& y){ const float n2=x*x+y*y; if(n2<=1e-20f) return false; const float inv=1.0f/std::sqrt(n2); x*=inv; y*=inv; return true; }
inline void rotateTowardsLimited(float& dirX,float& dirY,float tx,float ty,float maxAngle){
    if(!normalize2D(tx,ty)) return; if(!normalize2D(dirX,dirY)){ dirX=tx; dirY=ty; return; }
    const float dot=clampf(dirX*tx+dirY*ty,-1.0f,1.0f); const float ang=std::acos(dot);
    if(!(ang>0.0f)||ang<=maxAngle){ dirX=tx; dirY=ty; return; }
    const float crossZ=dirX*ty-dirY*tx; const float rot=(crossZ>=0.0f)?maxAngle:-maxAngle;
    const float c=std::cos(rot), s=std::sin(rot); const float nx=c*dirX - s*dirY; const float ny=s*dirX + c*dirY; dirX=nx; dirY=ny;
}
inline bool insideCardioidOrBulb(double x,double y) noexcept {
    const double xm=x-0.25; const double q=xm*xm + y*y;
    if(q * (q + xm) < 0.25 * y * y) return true;
    const double dx=x+1.0; if(dx*dx + y*y < 0.0625) return true;
    return false;
}
inline void antiVoidDriftNDC(float cx,float cy,float& ndcX,float& ndcY){
    float vx1=cx-0.25f, vy1=cy; float vx2=cx+1.0f, vy2=cy;
    float vx=0.5f*vx1 + 0.5f*vx2; float vy=0.5f*vy1 + 0.5f*vy2;
    if(!normalize2D(vx,vy)){ vx=1.0f; vy=0.0f; } ndcX=vx; ndcY=vy;
}
float median_inplace(std::vector<float>& v){
    if(v.empty()) return 0.0f; const size_t n=v.size(), mid=n/2; std::nth_element(v.begin(), v.begin()+mid, v.end());
    float m=v[mid]; if((n&1)==0){ std::nth_element(v.begin(), v.begin()+mid-1, v.begin()+mid); m=0.5f*(m+v[mid-1]); }
    return m;
}
float mad_from_center_inplace(std::vector<float>& v,float med){
    if(v.empty()) return 1.0f; for(float& x:v) x=std::fabs(x-med); float mad=median_inplace(v); return (mad>1e-6f)?mad:1.0f;
}
double centerDzdcMag(double cx,double cy,int maxIter=kM0_ITER){
    double zx=0.0,zy=0.0,dx=0.0,dy=0.0; const double cx0=cx, cy0=cy;
    for(int i=0;i<maxIter;++i){
        const double tdx=2.0*(zx*dx - zy*dy)+1.0, tdy=2.0*(zx*dy + zy*dx);
        dx=tdx; dy=tdy;
        const double nzx=zx*zx - zy*zy + cx0, nzy=2.0*zx*zy + cy0;
        zx=nzx; zy=nzy;
        if(zx*zx + zy*zy > 4.0) break;
    }
    return std::sqrt(dx*dx + dy*dy);
}
thread_local bool  g_dirInit=false;
thread_local float g_prevDirX=1.0f, g_prevDirY=0.0f;
} // namespace

namespace ZoomLogic {

float computeEntropyContrast(const std::vector<float>& entropy,int width,int height,int tileSize) noexcept {
    if(width<=0||height<=0||tileSize<=0) return 0.0f;
    const int tilesX=(width + tileSize - 1)/tileSize, tilesY=(height + tileSize - 1)/tileSize, total=tilesX*tilesY;
    if(total<=0 || (int)entropy.size()<total) return 0.0f;
    double acc=0.0; int cnt=0;
    for(int ty=0;ty<tilesY;++ty) for(int tx=0;tx<tilesX;++tx){
        const int i=ty*tilesX+tx; const float c=entropy[i]; float sum=0.0f; int n=0;
        const int nx[4]={tx-1,tx+1,tx,tx}; const int ny[4]={ty,ty,ty-1,ty+1};
        for(int k=0;k<4;++k){ if(nx[k]<0||ny[k]<0||nx[k]>=tilesX||ny[k]>=tilesY) continue; sum += std::fabs(entropy[ny[k]*tilesX+nx[k]]-c); ++n; }
        if(n>0){ acc += sum/n; ++cnt; }
    }
    return (cnt>0)? static_cast<float>(acc/cnt) : 0.0f;
}

ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    int tilesX, int tilesY,
    int width, int height,
    float2 currentOffset, float zoom,
    float2 previousOffset,
    ZoomState& state) noexcept
{
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    static clock::time_point s_last; static bool s_haveLast=false;
    double dt = s_haveLast ? std::chrono::duration<double>(t0 - s_last).count() : (1.0/60.0);
    s_last = t0; s_haveLast=true; dt = std::clamp(dt, 1.0/240.0, 1.0/15.0);

    static bool warmInit=false; static clock::time_point warmStart;
    if(!warmInit){ warmStart=t0; warmInit=true; }
    const bool freezeDirection = (std::chrono::duration<double>(t0 - warmStart).count() < Settings::warmUpFreezeSeconds);

    ZoomResult out{}; out.bestIndex=-1; out.shouldZoom=false; out.isNewTarget=false; out.newOffset=previousOffset; out.minDistance=0.02f;

    const int totalTiles = tilesX*tilesY;
    if(tilesX<=0||tilesY<=0||totalTiles<=0 || (int)entropy.size()<totalTiles || (int)contrast.size()<totalTiles){
        out.shouldZoom = Settings::ForceAlwaysZoom; return out; }

    // small micro-opts
    const double invWidth  = (width  > 0) ? (1.0 / double(width )) : 0.0;
    const double invHeight = (height > 0) ? (1.0 / double(height)) : 0.0;

    const double invZoom    = 1.0 / std::max(1e-6f, zoom);
    const double M0         = centerDzdcMag((double)currentOffset.x, (double)currentOffset.y, kM0_ITER);
    const double M0c        = std::log1p(std::max(0.0, M0));
    const double metricFac  = std::clamp(1.0 / (1.0 + (double)kMETRIC_ALPHA * M0c),
                                         (double)kMETRIC_MIN_F, (double)kMETRIC_MAX_F);
    const double invZoomEff = invZoom * metricFac;

    if (freezeDirection) {
        out.shouldZoom = true;
        if (insideCardioidOrBulb(currentOffset.x, currentOffset.y)) {
            float nx=1.0f, ny=0.0f; antiVoidDriftNDC(currentOffset.x, currentOffset.y, nx, ny);
            const float2 target = make_float2(previousOffset.x + nx * (float)(kWARMUP_DRIFT_NDC*invZoomEff),
                                              previousOffset.y + ny * (float)(kWARMUP_DRIFT_NDC*invZoomEff));
            const float a=0.20f;
            out.newOffset = make_float2(previousOffset.x*(1.0f-a)+target.x*a,
                                        previousOffset.y*(1.0f-a)+target.y*a);
        } else {
            float sx = g_dirInit? g_prevDirX : 1.0f; float sy = g_dirInit? g_prevDirY : 0.0f;
            const float2 target = make_float2(previousOffset.x + sx * (float)(kSEED_STEP_NDC*invZoomEff),
                                              previousOffset.y + sy * (float)(kSEED_STEP_NDC*invZoomEff));
            const float a=0.20f;
            out.newOffset = make_float2(previousOffset.x*(1.0f-a)+target.x*a,
                                        previousOffset.y*(1.0f-a)+target.y*a);
        }
        out.distance = std::hypot(out.newOffset.x-previousOffset.x, out.newOffset.y-previousOffset.y);
        return out;
    }

    // Median/MAD on working copies (reserve to avoid re-alloc churn)
    std::vector<float> e; e.reserve((size_t)totalTiles);
    std::vector<float> c; c.reserve((size_t)totalTiles);
    e.insert(e.end(), entropy.begin(), entropy.begin()+totalTiles);
    c.insert(c.end(), contrast.begin(), contrast.begin()+totalTiles);
    const float eMed = median_inplace(e); const float eMad = mad_from_center_inplace(e, eMed);
    const float cMed = median_inplace(c); const float cMad = mad_from_center_inplace(c, cMed);

    // Single pass for best score + moments
    float  bestScore = -1e9f; int bestIdx = -1;
    double meanS=0.0, meanS2=0.0;
    for (int i=0;i<totalTiles;++i){
        const float ez=(entropy[i]-eMed)/eMad;
        const float cz=(contrast[i]-cMed)/cMad;
        const float s = kALPHA_E*ez + kBETA_C*cz;
        meanS  += s;
        meanS2 += double(s)*s;
        if (s>bestScore){ bestScore=s; bestIdx=i; }
    }
    meanS/=std::max(1, totalTiles);
    const double varS = std::max(0.0, meanS2/std::max(1,totalTiles) - meanS*meanS);
    const double stdS = std::sqrt(varS);

    float temp = kTEMP_BASE; if (stdS>1e-6) temp = static_cast<float>(kTEMP_BASE/(0.5f+(float)stdS)); temp = clampf(temp, 0.2f, 2.5f);
    const float sCut = bestScore + temp * kSOFTMAX_LOG_EPS; const float invTemp = 1.0f/std::max(1e-6f,temp);

    // Softmax accumulation (recompute NDC on demand; identical numerics)
    double sumW=0.0, numX=0.0, numY=0.0; int bestAdj=-1; float bestAdjScore=-1e9f;
    for (int i=0;i<totalTiles;++i){
        const float ez=(entropy[i]-eMed)/eMad;
        const float cz=(contrast[i]-cMed)/cMad;
        const float s = kALPHA_E*ez + kBETA_C*cz;
        if (s<sCut) continue;

        auto p = tileIndexToPixelCenter(i, tilesX, tilesY, width, height);
        const double cx = double(p.first)  * invWidth;
        const double cy = double(p.second) * invHeight;
        const double ndcX = (cx-0.5)*2.0;
        const double ndcY = (cy-0.5)*2.0;

        const double tx = currentOffset.x + ndcX*invZoomEff;
        const double ty = currentOffset.y + ndcY*invZoomEff;
        if (insideCardioidOrBulb(tx,ty)) continue;

        const double w = std::exp(double((s - bestScore)*invTemp));
        sumW += w; numX += w*ndcX; numY += w*ndcY;
        if (s>bestAdjScore){ bestAdjScore=s; bestAdj=i; }
    }

    // Target selection (with exact same fallbacks)
    double ndcTX=0.0, ndcTY=0.0;
    if (sumW>0.0){
        const double inv = 1.0/sumW; ndcTX = numX*inv; ndcTY = numY*inv;
    } else if (bestAdj>=0){
        auto p = tileIndexToPixelCenter(bestAdj, tilesX, tilesY, width, height);
        ndcTX = (double(p.first )*invWidth  - 0.5)*2.0;
        ndcTY = (double(p.second)*invHeight - 0.5)*2.0;
    } else if (bestIdx>=0){
        auto p = tileIndexToPixelCenter(bestIdx, tilesX, tilesY, width, height);
        const double ndcBX = (double(p.first )*invWidth  - 0.5)*2.0;
        const double ndcBY = (double(p.second)*invHeight - 0.5)*2.0;
        const double tx = currentOffset.x + ndcBX*invZoomEff;
        const double ty = currentOffset.y + ndcBY*invZoomEff;
        if (!insideCardioidOrBulb(tx,ty)) { ndcTX=ndcBX; ndcTY=ndcBY; }
    }
    if (ndcTX==0.0 && ndcTY==0.0){
        ndcTX = g_dirInit? g_prevDirX : 1.0f; ndcTY = g_dirInit? g_prevDirY : 0.0f;
    }

    // ---- Patch A: minimal NDC-Target-Inertia (keine weiteren Änderungen) ----
    if (g_dirInit) {
        ndcTX = 0.7f * (float)ndcTX + 0.3f * g_prevDirX;
        ndcTY = 0.7f * (float)ndcTY + 0.3f * g_prevDirY;
    }
    // -------------------------------------------------------------------------

    const float2 rawTarget = make_float2(previousOffset.x + (float)(ndcTX*invZoomEff),
                                         previousOffset.y + (float)(ndcTY*invZoomEff));
    float mvx = rawTarget.x - previousOffset.x; float mvy = rawTarget.y - previousOffset.y;
    const float rawDist = std::sqrt(mvx*mvx + mvy*mvy);

    float dirX = g_dirInit? g_prevDirX : (rawDist>0.0f? mvx/rawDist : 1.0f);
    float dirY = g_dirInit? g_prevDirY : (rawDist>0.0f? mvy/rawDist : 0.0f); g_dirInit=true;
    float tgtX = mvx, tgtY = mvy; const bool hasMove = normalize2D(tgtX,tgtY);

    const float sigFactor  = clampf(static_cast<float>(stdS), 0.0f, 1.0f);
    const float distFactor = clampf(rawDist/0.25f, 0.0f, 1.0f);
    const float omega = kTURN_OMEGA_MIN + (kTURN_OMEGA_MAX - kTURN_OMEGA_MIN) * std::max(sigFactor, distFactor);
    const float maxTurn = omega * static_cast<float>(dt);

    float lenScale = 1.0f;
    if (hasMove){
        const float preDot = clampf(dirX*tgtX + dirY*tgtY, -1.0f, 1.0f);
        const float preAng = std::acos(preDot);
        rotateTowardsLimited(dirX, dirY, tgtX, tgtY, maxTurn);
        lenScale = 1.0f - smoothstepf(kTHETA_DAMP_LO, kTHETA_DAMP_HI, preAng);
        g_prevDirX = dirX; g_prevDirY = dirY;
    }

    float2 proposed = make_float2(previousOffset.x + dirX * (rawDist*lenScale),
                                  previousOffset.y + dirY * (rawDist*lenScale));

    // --- Nacktmull-Block entfernt: kein optionaler 3D-Planner mehr ---

    const float dist = std::hypot(proposed.x-previousOffset.x, proposed.y-previousOffset.y);
    const float distNorm = clampf(dist/0.5f, 0.0f, 1.0f);
    const float tau = kEMA_TAU_MAX + (kEMA_TAU_MIN - kEMA_TAU_MAX) * distNorm;
    float emaAlpha = 1.0f - std::exp(-static_cast<float>(dt)/std::max(1e-5f, tau));
    emaAlpha = clampf(emaAlpha, kEMA_ALPHA_MIN, kEMA_ALPHA_MAX);
    if (Settings::ForceAlwaysZoom && stdS < kMIN_SIGNAL_STD) emaAlpha = std::max(emaAlpha, kFORCE_MIN_DRIFT_ALPHA);

    const float2 smoothed = make_float2(previousOffset.x*(1.0f-emaAlpha) + proposed.x*emaAlpha,
                                        previousOffset.y*(1.0f-emaAlpha) + proposed.y*emaAlpha);

    const bool hasSignal = (stdS >= kMIN_SIGNAL_STD);
    out.shouldZoom = hasSignal || Settings::ForceAlwaysZoom;
    out.newOffset  = out.shouldZoom ? smoothed : previousOffset;
    out.distance   = std::hypot(out.newOffset.x-previousOffset.x, out.newOffset.y-previousOffset.y);

    // Hide interest markers in HUD:
    out.bestIndex = -1;
    out.isNewTarget = false;

    state.lastOffset = out.newOffset; state.lastTilesX = tilesX; state.lastTilesY = tilesY; state.cooldownLeft = 0;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM-LITE] invZoomEff=%.3g dist=%.6f ndc=%.6f len=%.3f ema=%.3f omega=%.3f",
               (float)invZoomEff,
               out.distance,
               (float)std::sqrt((double)ndcTX*(double)ndcTX + (double)ndcTY*(double)ndcTY),
               lenScale,
               emaAlpha,
               maxTurn);
    }
    return out;
}

} // namespace ZoomLogic
