///// Otter: Flow-Zoom V6 (Stats-Single-Path) — Interest-Guard + Softmax + Strafe-Kill + Heading-Lock + EMA.
///// Schneefuchs: MSVC-safe (std::max<double>), robuste Checks, klare Logs; keine Alternativpfade.
///// Maus: „wie im Fluss“ – zentrierte Zielwahl über Top-K-Schwerpunkt; kein Seitwärtswobble beim Zoomen.
///// Datei: src/zoom_logic.cpp

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
#include <utility>

#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace ZoomLogic {

// ============================== Tunables ====================================
namespace Flow {
    inline constexpr int   kFlatGuardFrames   = 12;
    inline constexpr float kFlatRelThreshold  = 0.02f;
    inline constexpr float kSoftmaxBeta       = 7.5f;
    inline constexpr float kLogEps            = -8.5f;   // sparsify (log-space)
    inline constexpr int   kTopK              = 6;

    inline constexpr float kEmaPanAlpha       = 0.30f;
    inline constexpr float kEmaZoomAlpha      = 0.25f;

    inline constexpr float kBaseZoomVel       = 0.018f;  // log-Skala
    inline constexpr float kBasePanVelNDC     = 0.045f;  // NDC/Frame

    inline constexpr float kStrafeK           = 60.0f;
    inline constexpr float kZoomRateLockMin   = 0.004f;

    inline constexpr int   kHeadingLockFrames = 36;      // ≈0.6s @60fps
    inline constexpr float kYawDegPerFrame    = 3.0f;
    inline constexpr float kPanDeadbandNDC    = 0.002f;
}

// =============================== State ======================================
struct FlowState {
    float     prevLogZoom = 0.0f;

    int       flatRun     = 0;
    int       lockFrames  = 0;
    float     lastHeading = 0.0f;

    float     emaPanX     = 0.0f;
    float     emaPanY     = 0.0f;
    float     emaZoom     = 0.0f;

    uint64_t  frameIdx    = 0;
};
static FlowState g;

// ============================== Helpers =====================================
static inline float clampf(float v, float a, float b){ return std::max(a, std::min(v, b)); }
static inline float radians(float deg){ return deg * 3.14159265358979323846f / 180.0f; }
static inline float atan2safe(float y, float x){ return std::atan2(y, x); }
static inline float wrap_pi(float a){ while(a>3.14159265f) a-=6.28318531f; while(a<-3.14159265f) a+=6.28318531f; return a; }

static std::pair<int,float> argmax2(const std::vector<float>& s){
    int bi=-1; float bv=-std::numeric_limits<float>::infinity();
    int si=-1; float sv=-std::numeric_limits<float>::infinity();
    for(size_t i=0;i<s.size();++i){
        float v=s[i];
        if(v>bv){ si=bi; sv=bv; bi=(int)i; bv=v; }
        else if(v>sv){ si=(int)i; sv=v; }
    }
    return {bi, sv};
}

static void sort_topk(const std::vector<float>& s, int k, std::vector<int>& outIdx){
    outIdx.clear(); outIdx.reserve((size_t)k);
    outIdx.push_back(-1);
    auto [bi,_sv] = argmax2(s);
    int si=-1;
    if(bi>=0){
        float sv=-std::numeric_limits<float>::infinity();
        for(size_t i=0;i<s.size();++i){
            if((int)i==bi) continue;
            float v=s[i]; if(v>sv){ sv=v; si=(int)i; }
        }
    }
    if(bi>=0) outIdx[0]=bi;
    if(si>=0) outIdx.push_back(si);
    for(size_t i=0;i<s.size();++i){
        if((int)i==bi || (int)i==si) continue;
        outIdx.push_back((int)i);
    }
    std::partial_sort(outIdx.begin(), outIdx.begin()+std::min<int>(k,(int)outIdx.size()), outIdx.end(),
                      [&](int a,int b){ return s[(size_t)a] > s[(size_t)b]; });
    if((int)outIdx.size()>k) outIdx.resize((size_t)k);
}

static inline void indexToXY(int idx,int tilesX,int& x,int& y){ x = (idx % tilesX); y = (idx / tilesX); }
static inline void tileCenterNDC(int tx,int ty,int tilesX,int tilesY,float& nx,float& ny){
    const float cx=(float(tx)+0.5f)/float(tilesX), cy=(float(ty)+0.5f)/float(tilesY);
    nx = cx*2.0f - 1.0f;
    ny = 1.0f     - cy*2.0f;
}

// ============================ Public Helpers =================================
float computeEntropyContrast(const std::vector<float>& entropy,
                             int width, int height, int tileSize) noexcept
{
    if(width<=0 || height<=0 || tileSize<=0) return 0.0f;
    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int n      = tilesX * tilesY;
    if((int)entropy.size() < n) return 0.0f;

    double acc = 0.0;
    int    cnt = 0;
    auto at = [&](int x,int y)->float{ return entropy[(size_t)(y*tilesX + x)]; };

    for(int y=0;y<tilesY;++y){
        for(int x=0;x<tilesX;++x){ // FIX: x < tilesX (nicht tilesY)
            const float c = at(x,y);
            if(x+1<tilesX){ acc += std::fabs(c - at(x+1,y)); ++cnt; }
            if(y+1<tilesY){ acc += std::fabs(c - at(x,y+1)); ++cnt; }
        }
    }
    if(cnt==0) return 0.0f;

    if constexpr (Settings::debugLogging) {
        const int expected = (tilesX - 1) * tilesY + (tilesY - 1) * tilesX; // 2XY - X - Y
        if (cnt != expected) {
            LUCHS_LOG_HOST("[Zoom/Entropy] neighbor-pairs mismatch: cnt=%d expected=%d tiles=%dx%d",
                           cnt, expected, tilesX, tilesY);
        }
    }
    return static_cast<float>(acc / (double)cnt);
}

ZoomResult evaluateZoomTarget(const std::vector<float>& entropy,
                              const std::vector<float>& contrast,
                              int tilesX, int tilesY,
                              int width, int height,
                              float2 currentOffset, float zoom,
                              float2 previousOffset,
                              ZoomState& state) noexcept
{
    ZoomResult zr{};
    const int nTiles = tilesX * tilesY;
    if(nTiles<=0 || (int)entropy.size()<nTiles || (int)contrast.size()<nTiles){
        state.hadCandidate = false;
        zr.shouldZoom = false;
        zr.bestIndex  = -1;
        zr.newOffsetX = currentOffset.x;
        zr.newOffsetY = currentOffset.y;
        return zr;
    }

    // Score = Entropy + Contrast
    std::vector<float> score; score.resize((size_t)nTiles);
    float maxV = 1e-9f;
    for(int i=0;i<nTiles;++i){
        float v = entropy[(size_t)i] + contrast[(size_t)i];
        score[(size_t)i]=v; if(v>maxV) maxV=v;
    }
    const float bestV = *std::max_element(score.begin(), score.end());
    state.hadCandidate = (bestV > 0.0f);

    // Top-K Softmax-Schwerpunkt (wie Flow-Pfad, ohne Zeitbezug)
    std::vector<int> top; sort_topk(score, Flow::kTopK, top);
    float accX=0.0f, accY=0.0f, accW=0.0f;
    const float maxTop = (top.empty()? 0.0f : score[(size_t)top[0]]);
    for(size_t i=0;i<top.size();++i){
        const int idx = top[i];
        float w = score[(size_t)idx];
        float z = (w - maxTop) + Flow::kLogEps;
        float sw = std::exp(Flow::kSoftmaxBeta * z);
        int tx,ty; indexToXY(idx, tilesX, tx, ty);
        float nx,ny; tileCenterNDC(tx,ty,tilesX,tilesY,nx,ny);
        accX += sw * nx; accY += sw * ny; accW += sw;
    }

    float dirX=0.0f, dirY=0.0f;
    if(accW > 0.0f){ dirX = accX/accW; dirY = accY/accW; }
    const float len = std::sqrt(dirX*dirX + dirY*dirY);
    if(len>1e-6f){ dirX/=len; dirY/=len; }

    // Schrittgröße heuristisch von Zoom abhängig (log-Skala, kein Δt hier)
    const double zlog = std::log(std::max<double>(1e-12, (double)zoom));
    const float  step = Flow::kBasePanVelNDC * (0.25f + 0.75f) * (1.0f); // konservativ

    zr.newOffsetX = currentOffset.x + dirX * step;
    zr.newOffsetY = currentOffset.y + dirY * step;
    zr.distance   = std::sqrt((zr.newOffsetX - previousOffset.x)*(zr.newOffsetX - previousOffset.x) +
                              (zr.newOffsetY - previousOffset.y)*(zr.newOffsetY - previousOffset.y));
    zr.minDistance= 0.02f;
    zr.bestIndex  = top.empty()? -1 : top[0];
    zr.isNewTarget= false; // Ohne Historie des besten Index hier neutral lassen
    zr.shouldZoom = state.hadCandidate;

    (void)width; (void)height; (void)zlog; // aktuell ungenutzt, API-kompatibel
    return zr;
}

// ================================ Core ======================================
static void update(FrameContext& frameCtx, RendererState& rs, ZoomState& zs)
{
    g.frameIdx++;

    // Geometrie
    const int width  = frameCtx.width;
    const int height = frameCtx.height;
    if (width <= 0 || height <= 0) { zs.hadCandidate = false; return; }

    // Analyse-Raster (entkoppelt): statsTileSize bevorzugen, sonst Fallback auf Compute-Raster.
    const int statsPx  = (frameCtx.statsTileSize > 0 ? frameCtx.statsTileSize : frameCtx.tileSize);
    if (statsPx <= 0) { zs.hadCandidate = false; return; }

    const int tilesX = (width  + statsPx - 1) / statsPx;
    const int tilesY = (height + statsPx - 1) / statsPx;
    const int nTiles = tilesX * tilesY;

    // Puffergrößen müssen zum Stats-Raster passen.
    if ((int)frameCtx.entropy.size()  < nTiles ||
        (int)frameCtx.contrast.size() < nTiles) {
        zs.hadCandidate = false;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[Zoom/Flow] ERROR stats size mismatch: need=%d e=%zu c=%zu statsPx=%d",
                           nTiles, frameCtx.entropy.size(), frameCtx.contrast.size(), statsPx);
        }
        return;
    }

    // 1) Score + Max
    std::vector<float> score; score.resize((size_t)nTiles);
    float maxV = 1e-9f;
    for (int i = 0; i < nTiles; ++i) {
        float v = frameCtx.entropy[(size_t)i] + frameCtx.contrast[(size_t)i];
        score[(size_t)i] = v; if (v > maxV) maxV = v;
    }
    const float bestV = *std::max_element(score.begin(), score.end());
    const bool  isFlat = (bestV < Flow::kFlatRelThreshold * maxV) || (bestV <= 0.0f);
    if (isFlat) g.flatRun++; else g.flatRun = 0;
    zs.hadCandidate = (bestV > 0.0f);

    // 2) Top-K Softmax-Schwerpunkt
    std::vector<int> top; sort_topk(score, Flow::kTopK, top);

    if (g.flatRun >= Flow::kFlatGuardFrames && (int)top.size() >= 2) {
        std::swap(top[0], top[1]);                 // „zweitbestes“ Ziel erzwingen
        g.lockFrames = Flow::kHeadingLockFrames;   // Kurs nach Retarget sanft begrenzen
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[Zoom/Flow] retarget second best after flat=%d", g.flatRun);
        }
        g.flatRun = 0;
    }

    float accX=0.0f, accY=0.0f, accW=0.0f;
    const float maxTop = (top.empty()? 0.0f : score[(size_t)top[0]]);
    const float logEps = Flow::kLogEps;            // sparsify im Log-Raum
    for (size_t i = 0; i < top.size(); ++i) {
        const int idx = top[i];
        float w = score[(size_t)idx];
        float z = (w - maxTop) + logEps;           // -> stärkerer Fokus auf Bestes
        float sw = std::exp(Flow::kSoftmaxBeta * z);
        int tx,ty; indexToXY(idx, tilesX, tx, ty);
        float nx,ny; tileCenterNDC(tx,ty,tilesX,tilesY,nx,ny);
        accX += sw * nx; accY += sw * ny; accW += sw;
    }

    float tgtX = 0.0f, tgtY = 0.0f;
    if (accW > 0.0f) { tgtX = accX/accW; tgtY = accY/accW; }

    // 3) Richtung (NDC) normalisieren
    float panX = tgtX, panY = tgtY;
    const float panLen = std::sqrt(panX*panX + panY*panY);
    if (panLen > 1e-6f) { panX/=panLen; panY/=panLen; }

    // 4) Strafe-Kill (Seitwärtsdämpfung bei starkem Zoom)
    const double zoomForLog = std::max<double>(1e-12, static_cast<double>(RS_ZOOM(rs)));
    const float  logZoom    = static_cast<float>(std::log(zoomForLog));
    const float  dLogZoom   = logZoom - g.prevLogZoom;
    g.prevLogZoom           = logZoom;

    float strafeFactor = 1.0f - Flow::kStrafeK * std::fabs(dLogZoom);
    if (std::fabs(dLogZoom) >= Flow::kZoomRateLockMin) strafeFactor = clampf(strafeFactor, 0.0f, 0.1f);
    else                                               strafeFactor = clampf(strafeFactor, 0.0f, 1.0f);

    // 5) Pan-Speed (inhaltsgewichtet) + Deadband
    const float contentW = clampf(maxV>0.0f ? (bestV/maxV) : 0.0f, 0.15f, 1.0f);
    float panSpeed = Flow::kBasePanVelNDC * (0.25f + 0.75f * contentW);
    panSpeed *= strafeFactor;

    if (panLen < Flow::kPanDeadbandNDC) { panX=0.0f; panY=0.0f; panSpeed=0.0f; }

    // 6) Heading-Lock (sanfter Kurs nach Retarget)
    if (g.lockFrames > 0){
        float curHeading = atan2safe(panY, panX);
        float delta      = wrap_pi(curHeading - g.lastHeading);
        const float maxDelta = radians(Flow::kYawDegPerFrame);
        if (std::fabs(delta) > maxDelta){
            const float sign = (delta>0.0f? 1.0f : -1.0f);
            curHeading = g.lastHeading + sign * maxDelta;
            panX = std::cos(curHeading);
            panY = std::sin(curHeading);
        }
        g.lastHeading = curHeading;
        g.lockFrames--;
    } else {
        g.lastHeading = atan2safe(panY, panX);
    }

    // 7) EMA-Glättung
    g.emaPanX = (1.0f - Flow::kEmaPanAlpha) * g.emaPanX + Flow::kEmaPanAlpha * (panX * panSpeed);
    g.emaPanY = (1.0f - Flow::kEmaPanAlpha) * g.emaPanY + Flow::kEmaPanAlpha * (panY * panSpeed);

    const float zoomVel = Flow::kBaseZoomVel * contentW;
    g.emaZoom  = (1.0f - Flow::kEmaZoomAlpha) * g.emaZoom + Flow::kEmaZoomAlpha * zoomVel;

    // 8) Zeitnormierung
    const float dt = (frameCtx.deltaSeconds > 0.0f ? frameCtx.deltaSeconds : 1.0f/60.0f);

    // 9) Anwenden
    RS_OFFSET_X(rs) += g.emaPanX * dt;
    RS_OFFSET_Y(rs) += g.emaPanY * dt;
    RS_ZOOM(rs)     *= std::exp(g.emaZoom * dt);

    // ---- Flow Telemetry (Einzeiler) ----
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST(
            "[FlowTL] f=%llu statsPx=%d n=%d best=%.4f max=%.4f cand=%d pan=(%.4f,%.4f) zoom=%.6f",
            (unsigned long long)g.frameIdx, statsPx, nTiles, bestV, maxV,
            zs.hadCandidate ? 1 : 0, g.emaPanX, g.emaPanY, RS_ZOOM(rs)
        );
    }

    if constexpr(Settings::performanceLogging){
        LUCHS_LOG_HOST("[Zoom/Flow] f=%llu flat=%d best=%.4f max=%.4f dlog=%.5f strafe=%.3f pan=(%.4f,%.4f) pspd=%.4f zoom=%.6f statsPx=%d",
                       (unsigned long long)g.frameIdx, g.flatRun, bestV, maxV, dLogZoom, strafeFactor,
                       g.emaPanX, g.emaPanY, panSpeed, RS_ZOOM(rs), statsPx);
    }
}

// ---------------------------------------------------------------------------
// Legacy/Kompatibilität: erwarteter Entry aus frame_pipeline.cpp
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
