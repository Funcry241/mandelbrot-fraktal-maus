///// Otter: Flow-Zoom V6 — orbit-killer (tangential damp), heading-lock, dir-anchor; no retarget swaps.
///** Schneefuchs: MSVC /WX clean; remove unused code; ASCII logs only; API unchanged.
///** Maus: sanft, kreisfest; dt-clamp; EMA-Bewegung ohne Mikrowobble.
///** Datei: src/zoom_logic.cpp

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

#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace ZoomLogic {

// ============================== Tunables ====================================
namespace Flow {
    inline constexpr int   kFlatGuardFrames     = 12;
    inline constexpr float kFlatRelThreshold    = 0.02f;

    // Sanftere Top-K-Gewichtung (stabilere Schwerpunkte)
    inline constexpr float kSoftmaxBeta         = 6.3f;
    inline constexpr float kLogEps              = -7.3f;   // sparsify (log-space)
    inline constexpr int   kTopK                = 6;

    inline constexpr float kEmaPanAlpha         = 0.30f;
    inline constexpr float kEmaZoomAlpha        = 0.25f;

    inline constexpr float kBaseZoomVel         = 0.18f;   // log-scale
    inline constexpr float kBasePanVelNDC       = 0.045f;  // NDC/frame

    // Weiche (nicht binäre) Strafe-Charakteristik
    inline constexpr float kStrafeKSoft         = 30.0f;   // aggressiveness in 1/(1+k*|dlog|)
    inline constexpr float kStrafeMin           = 0.35f;   // niemals komplett abwürgen

    inline constexpr int   kHeadingLockFrames   = 36;      // ≈0.6s @60fps
    inline constexpr float kYawDegPerFrame      = 3.0f;    // max Rotationsrate unter Lock
    inline constexpr float kHeadingTriggerDeg   = 7.0f;    // ab diesem Sprung Lock aktivieren

    // Größere Deadband gegen Mikrowobble + Distanzskalierung
    inline constexpr float kPanDeadbandNDC      = 0.0040f;
    inline constexpr float kPanScaleNDC         = 0.0800f; // Obergrenze für smoothstep-Skalierung

    // Richtungs-Anker gegen Kreisdrehen
    inline constexpr float kDirEmaAlpha        = 0.22f;   // 0..1 — wie klebrig der Zielvektor ist
    inline constexpr float kDirStick           = 0.65f;   // 0..1 — Mischung Anker vs. frische Richtung

    // Sanfte Dämpfung der tangentialen Komponente (Orbit-Killer)
    inline constexpr float kTangentialDamp     = 0.50f;   // 0..1 — 0 = aus, 1 = volle Projektion
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

    // Richtungsanker (NDC)
    float     anchX       = 0.0f;
    float     anchY       = 0.0f;

    uint64_t  frameIdx    = 0;
    int       prevCand    = -1;
};
static FlowState g;

// ============================== Helpers =====================================
static inline float clampf(float v, float a, float b){ return std::max(a, std::min(v, b)); }
static inline float radians(float deg){ return deg * 3.14159265358979323846f / 180.0f; }
static inline float atan2safe(float y, float x){ return std::atan2(y, x); }
static inline float wrap_pi(float a){ while(a>3.14159265f) a-=6.28318531f; while(a<-3.14159265f) a+=6.28318531f; return a; }
static inline float smoothstepf(float a, float b, float x){
    if (b <= a) return 0.0f;
    const float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// sort indices by top-k values without allocating weights repeatedly
static void sort_topk(const std::vector<float>& s, int k, std::vector<int>& outIdx){
    outIdx.clear(); outIdx.reserve(s.size());
    // seed with {best, second-best} in O(n), then partial_sort
    int bi=-1; float bv=-std::numeric_limits<float>::infinity();
    int si=-1; float sv=-std::numeric_limits<float>::infinity();
    for(size_t i=0;i<s.size();++i){
        const float v=s[i];
        if(v>bv){ si=bi; sv=bv; bi=(int)i; bv=v; }
        else if(v>sv && (int)i!=bi){ si=(int)i; sv=v; }
    }
    if(bi>=0) outIdx.push_back(bi);
    if(si>=0) outIdx.push_back(si);
    for(size_t i=0;i<s.size();++i){
        if((int)i==bi || (int)i==si) continue;
        outIdx.push_back((int)i);
    }
    const int want = std::min<int>(k, (int)outIdx.size());
    std::partial_sort(outIdx.begin(), outIdx.begin()+want, outIdx.end(),
                      [&](int a,int b){ return s[(size_t)a] > s[(size_t)b]; });
    if((int)outIdx.size()>k) outIdx.resize((size_t)k);
}

// map tile index to (tx,ty)
static inline void indexToXY(int idx, int tilesX, int& tx, int& ty){
    tx = (idx % tilesX);
    ty = (idx / tilesX);
}

// center of a tile in NDC (-1..+1)
static inline void tileCenterNDC(int tx, int ty, int tilesX, int tilesY, float& nx, float& ny){
    const float x0 = (tx + 0.5f) / (float)tilesX;
    const float y0 = (ty + 0.5f) / (float)tilesY;
    nx = 2.0f*x0 - 1.0f;
    ny = 2.0f*y0 - 1.0f;
}

// ===========================================================================
// Öffentliche Kern-API (Score → Zielrichtung → Glättung → Anwenden)
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
    const int nTiles = tilesX * tilesY;
    if(nTiles<=0 || (int)entropy.size()<nTiles || (int)contrast.size()<nTiles){
        state.hadCandidate = false;
        zr.shouldZoom = false;
        zr.bestIndex  = -1;
        zr.newOffsetX = currentOffset.x;
        zr.newOffsetY = currentOffset.y;
        return zr;
    }

    static std::vector<float> score;
    score.resize((size_t)nTiles);

    float maxV = 1e-9f;
    for(int i=0;i<nTiles;++i){
        const float v = entropy[(size_t)i] + contrast[(size_t)i];
        score[(size_t)i]=v; if(v>maxV) maxV=v;
    }
    const float bestV = maxV;
    state.hadCandidate = (bestV > 0.0f);

    std::vector<int> top; sort_topk(score, Flow::kTopK, top);
    float accX=0.0f, accY=0.0f, accW=0.0f;
    const float maxTop = (top.empty()? 0.0f : score[(size_t)top[0]]);
    for(size_t i=0;i<top.size();++i){
        const int idx = top[i];
        const float w = score[(size_t)idx];
        const float z = (w - maxTop) + Flow::kLogEps;
        const float sw = std::exp(Flow::kSoftmaxBeta * z);
        int tx,ty; indexToXY(idx, tilesX, tx, ty);
        float nx,ny; tileCenterNDC(tx,ty,tilesX,tilesY,nx,ny);
        accX += sw * nx; accY += sw * ny; accW += sw;
    }

    float dirX=0.0f, dirY=0.0f;
    if(accW > 0.0f){ dirX = accX/accW; dirY = accY/accW; }

    zr.bestIndex  = (top.empty()? -1 : top[0]);
    zr.shouldZoom = (bestV > 0.0f);
    // Nur Richtung melden; Offset-Anwendung erfolgt in evaluateAndApply()
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

    const int statsPx  = (frameCtx.statsTileSize > 0 ? frameCtx.statsTileSize : frameCtx.tileSize);
    if (statsPx <= 0) { zs.hadCandidate = false; return; }

    const int tilesX = (width  + statsPx - 1) / statsPx;
    const int tilesY = (height + statsPx - 1) / statsPx;
    const int nTiles = tilesX * tilesY;

    // Sicherheitsnetz: genügend Stats?
    if ((int)frameCtx.entropy.size() < nTiles || (int)frameCtx.contrast.size() < nTiles) {
        zs.hadCandidate = false;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[Zoom/Flow] ERROR stats size mismatch: need=%d e=%zu c=%zu statsPx=%d",
                           nTiles, frameCtx.entropy.size(), frameCtx.contrast.size(), statsPx);
        }
        return;
    }

    static std::vector<float> score;
    score.resize((size_t)nTiles);

    float maxV = 1e-9f;
    for (int i = 0; i < nTiles; ++i){
        const float v = frameCtx.entropy[(size_t)i] + frameCtx.contrast[(size_t)i];
        score[(size_t)i]=v; if(v>maxV) maxV=v;
    }
    const float bestV = maxV;
    const bool  isFlat = (bestV <= 0.0f) || (bestV < Flow::kFlatRelThreshold * maxV);
    if (isFlat) g.flatRun++; else g.flatRun = 0;
    zs.hadCandidate = (bestV > 0.0f);

    // *** HITCH-FIX: Retargeting vollständig deaktivieren ***
    // (flatRun bleibt nur fuer Telemetrie.)

    // Schwerpunkt aus Top-K via Softmax
    std::vector<int> top; sort_topk(score, Flow::kTopK, top);

    float accX=0.0f, accY=0.0f, accW=0.0f;
    const float maxTop = (top.empty()? 0.0f : score[(size_t)top[0]]);
    for (size_t i = 0; i < top.size(); ++i) {
        const int idx = top[i];
        const float w = score[(size_t)idx];
        const float z = (w - maxTop) + Flow::kLogEps;
        const float sw = std::exp(Flow::kSoftmaxBeta * z);
        int tx,ty; indexToXY(idx, tilesX, tx, ty);
        float nx,ny; tileCenterNDC(tx,ty,tilesX,tilesY,nx,ny);
        accX += sw * nx; accY += sw * ny; accW += sw;
    }

    float tgtX = 0.0f, tgtY = 0.0f;
    if (accW > 0.0f) { tgtX = accX/accW; tgtY = accY/accW; }

    // Distanz-gekoppelte Pan-Berechnung
    float panDirX = 0.0f, panDirY = 0.0f;
    float panLen  = std::sqrt(tgtX*tgtX + tgtY*tgtY);
    if (panLen > 1e-6f) {
        panDirX = tgtX / panLen;
        panDirY = tgtY / panLen;
    }

    // Richtungsanker: glättet die Zielrichtung ueber mehrere Frames (stoppt Kreisdrehen)
    if (panLen > 1e-6f) {
        const float ax = panDirX;
        const float ay = panDirY;
        const float a  = Flow::kDirEmaAlpha;
        g.anchX = (1.0f - a) * g.anchX + a * ax;
        g.anchY = (1.0f - a) * g.anchY + a * ay;
        const float al = std::sqrt(g.anchX*g.anchX + g.anchY*g.anchY);
        if (al > 1e-6f) {
            float dirAX = g.anchX / al;
            float dirAY = g.anchY / al;
            // Mischung aus frischer Richtung und Anker
            float mixX = (1.0f - Flow::kDirStick) * panDirX + Flow::kDirStick * dirAX;
            float mixY = (1.0f - Flow::kDirStick) * panDirY + Flow::kDirStick * dirAY;
            const float ml = std::sqrt(mixX*mixX + mixY*mixY);
            if (ml > 1e-6f) { panDirX = mixX / ml; panDirY = mixY / ml; }
        }
    }

    // Weiche Strafe (keine Abwürge-Plateaus)
    const double zoomForLog = std::max<double>(1e-12, static_cast<double>(RS_ZOOM(rs)));
    const float  logZoom    = static_cast<float>(std::log(zoomForLog));
    const float  dLogZoom   = logZoom - g.prevLogZoom;
    g.prevLogZoom           = logZoom;

    float strafeFactor = 1.0f / (1.0f + Flow::kStrafeKSoft * std::fabs(dLogZoom));
    strafeFactor = clampf(strafeFactor, Flow::kStrafeMin, 1.0f);

    // Inhaltsgewicht (derzeit ~1.0, belassen) + Distanz-Skalierung via smoothstep
    const float contentW = 1.0f;
    float panSpeed = Flow::kBasePanVelNDC * (0.25f + 0.75f * contentW);
    panSpeed *= strafeFactor;

    // Deadband + weiche Skalierung abhängig von der Zielentfernung
    float panScale = 0.0f;
    if (panLen > Flow::kPanDeadbandNDC) {
        panScale = smoothstepf(Flow::kPanDeadbandNDC, Flow::kPanScaleNDC, panLen);
    } else {
        panDirX = 0.0f; panDirY = 0.0f; // Richtung ignorieren innerhalb Deadband
    }

    // Heading-Lock: bei großen Richtungswechseln aktivieren
    if (panLen > 1e-6f) {
        const float rawHeading = atan2safe(panDirY, panDirX);
        const float delta      = wrap_pi(rawHeading - g.lastHeading);
        if (g.lockFrames == 0 && std::fabs(delta) > radians(Flow::kHeadingTriggerDeg)) {
            g.lockFrames = Flow::kHeadingLockFrames;
        }
    }

    // Orbit-Dämpfung: tangentiale Komponente relativ zur Ausgaberichtung reduzieren
    float outDirX = panDirX, outDirY = panDirY;
    if (panLen > 1e-6f) {
        const float px = -outDirY; // Perp zu outDir
        const float py =  outDirX;
        const float tang = g.emaPanX * px + g.emaPanY * py;
        const float corr = -Flow::kTangentialDamp * tang;
        outDirX += corr * px;
        outDirY += corr * py;
        const float ol = std::sqrt(outDirX*outDirX + outDirY*outDirY);
        if (ol > 1e-6f) { outDirX /= ol; outDirY /= ol; }
    }

    // Heading-Lock anwenden (max Rotationsrate)
    if (g.lockFrames > 0 && (panLen > 1e-6f)) {
        float curHeading = atan2safe(outDirY, outDirX);
        float delta      = wrap_pi(curHeading - g.lastHeading);
        const float maxDelta = radians(Flow::kYawDegPerFrame);
        if (std::fabs(delta) > maxDelta){
            const float sign = (delta>0.0f? 1.0f : -1.0f);
            curHeading = g.lastHeading + sign * maxDelta;
            outDirX = std::cos(curHeading);
            outDirY = std::sin(curHeading);
        }
        g.lastHeading = atan2safe(outDirY, outDirX);
        g.lockFrames--;
    } else if (panLen > 1e-6f) {
        g.lastHeading = atan2safe(panDirY, panDirX);
    }

    // EMA-Glättung (mit Distanz-Skalierung)
    const float effPan = panSpeed * panScale;
    g.emaPanX = (1.0f - Flow::kEmaPanAlpha) * g.emaPanX + Flow::kEmaPanAlpha * (outDirX * effPan);
    g.emaPanY = (1.0f - Flow::kEmaPanAlpha) * g.emaPanY + Flow::kEmaPanAlpha * (outDirY * effPan);

    const float zoomVel = Flow::kBaseZoomVel * contentW;
    g.emaZoom  = (1.0f - Flow::kEmaZoomAlpha) * g.emaZoom + Flow::kEmaZoomAlpha * zoomVel;

    // Zeitnormierung (clamp gegen Jitter)
    const float dtRaw = (frameCtx.deltaSeconds > 0.0f ? frameCtx.deltaSeconds : 1.0f/60.0f);
    const float dt    = clampf(dtRaw, 1.0f/200.0f, 1.0f/30.0f);

    // Anwenden
    RS_OFFSET_X(rs) += g.emaPanX * dt;
    RS_OFFSET_Y(rs) += g.emaPanY * dt;
    RS_ZOOM(rs)     *= std::exp(g.emaZoom * dt);

    // Schlanke Telemetrie (ASCII only)
    if constexpr (Settings::performanceLogging) {
        if ((g.frameIdx & 0xF) == 0) {
            LUCHS_LOG_HOST("[FlowTL] f=%llu statsPx=%d n=%d best=%.4f pan=(%.4f,%.4f) zoom=%.6f",
                           (unsigned long long)g.frameIdx, statsPx, nTiles, bestV,
                           g.emaPanX, g.emaPanY, RS_ZOOM(rs));
        }
        const int cand = zs.hadCandidate ? 1 : 0;
        if (g.prevCand != cand) {
            LUCHS_LOG_HOST("[FlowCAND] cand=%d statsPx=%d", cand, statsPx);
            g.prevCand = cand;
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
