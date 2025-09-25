///// Otter: Silk-Lite auto-pan/zoom controller with robust drift fallback (no heatmap required).
///// Schneefuchs: Keine API-Drifts; deterministische Schrittweite; MSVC-/WX-fest.
///// Maus: Sanfter Drift in NDC, Limitierung der Kurvenrate; ASCII-only.
///// Datei: src/zoom_logic.cpp

#include "zoom_logic.hpp"
#include "frame_context.hpp"    // definiert FrameContext (struct)
#include "renderer_state.hpp"   // definiert RendererState (class)
#include "cuda_interop.hpp"     // CudaInterop::getPauseZoom()
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

namespace {

// ----------------------------- Tunables ---------------------------------
constexpr float kSEED_STEP_NDC = 0.015f; // Basisschritt in NDC
constexpr float kSTEP_MAX_NDC  = 0.35f;  // Sicherheitskappe
constexpr float kTURN_MAX_RAD  = 0.18f;  // max. Richtungsänderung/Frame
constexpr float kBLEND_A       = 0.22f;  // Low-Pass fürs Offset

constexpr float kALPHA_E = 1.0f; // Gewicht Entropy
constexpr float kBETA_C  = 0.5f; // Gewicht Contrast

// Minimaler „Signifikanz“-Score (Z-Skala). Unterhalb → Drift-Fallback.
constexpr float kMIN_SCORE_SIGMA = 0.35f;

// Approx. Sichtspanne wie in cuda_interop (Fallback, falls pixelScale unbekannt)
constexpr double kBASE_SPAN = 8.0 / 3.0;

inline float clampf(float x, float a, float b) { return (x < a) ? a : (x > b ? b : x); }

inline bool normalize2D(float& x, float& y) {
    const float n2 = x*x + y*y;
    if (!(n2 > 0.0f)) return false;
    const float inv = 1.0f/std::sqrt(n2);
    x *= inv; y *= inv; return true;
}

inline void rotateTowardsLimited(float& dx, float& dy, float tx, float ty, float maxAngle) {
    if (!normalize2D(tx, ty)) return;
    if (!normalize2D(dx, dy)) { dx = tx; dy = ty; return; }
    const float dot = clampf(dx*tx + dy*ty, -1.0f, 1.0f);
    const float ang = std::sqrt(std::max(0.0f, 2.0f*(1.0f - dot))); // ≈ Winkel
    if (!(ang > 0.0f) || ang <= maxAngle) { dx = tx; dy = ty; return; }
    const float t = clampf(maxAngle/ang, 0.0f, 1.0f);
    float nx = (1.0f - t)*dx + t*tx;
    float ny = (1.0f - t)*dy + t*ty;
    if (!normalize2D(nx, ny)) { nx = tx; ny = ty; }
    dx = nx; dy = ny;
}

// Hauptmengen Test (Cardioid + Periode-2-Bulb)
inline bool insideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q*(q + xm) < 0.25*y*y) return true;
    const double dx = x + 1.0;
    return (dx*dx + y*y) < 0.0625;
}

// simpler Auswärts-Drift aus Bulbs
inline void antiVoidDriftNDC(float /*cx*/, float /*cy*/, float& outx, float& outy) {
    outx = 1.0f; outy = 0.0f; normalize2D(outx, outy);
}

// Median/MAD (in-place)
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
    return (mad > 1e-6f) ? mad : 1.0f; // Guard gegen degenerierte Verteilungen
}

// Tileindex → NDC-Zentrum (−1..+1)
inline void tileIndexToNdcCenter(int tilesX, int tilesY, int idx, float& ndcX, float& ndcY) {
    const int tx = (tilesX > 0) ? (idx % tilesX) : 0;
    const int ty = (tilesX > 0) ? (idx / tilesX) : 0;
    const float cx = (static_cast<float>(tx) + 0.5f) / std::max(1, tilesX);
    const float cy = (static_cast<float>(ty) + 0.5f) / std::max(1, tilesY);
    ndcX = cx * 2.0f - 1.0f;
    ndcY = cy * 2.0f - 1.0f;
}

// Richtungs-Gedächtnis
thread_local bool  g_dirInit  = false;
thread_local float g_prevDirX = 1.0f;
thread_local float g_prevDirY = 0.0f;

// ---------------------- NDC → Welt-Schrittfaktor ------------------------
// Welt/Px ≈ (kBASE_SPAN / width) * (1/zoom)  → Welt/NDC = Welt/Px * (width/2)
// Analog für Y mit height. Das reicht als robustes Approx, wenn pixelScale
// hier nicht verfügbar ist.
inline void ndcToWorldScales(int width, int height, float zoom, double& wx, double& wy) {
    const double invZ = 1.0 / std::max(1e-6, (double)zoom);
    const double stepPx = kBASE_SPAN / std::max(1, width); // Welt pro Pixel @ zoom=1
    const double stepNow = stepPx * invZ;                  // Welt pro Pixel @ zoom
    wx = stepNow * (0.5 * (double)width);
    wy = stepNow * (0.5 * (double)height);
}

} // namespace

namespace ZoomLogic {

float computeEntropyContrast(const std::vector<float>& entropy,
                             int width, int height, int tileSize) noexcept
{
    if (width <= 0 || height <= 0 || tileSize <= 0) return 0.0f;
    const int txs = (width  + tileSize - 1) / tileSize;
    const int tys = (height + tileSize - 1) / tileSize;
    const int total = txs * tys;
    if (total <= 0 || static_cast<int>(entropy.size()) < total) return 0.0f;

    double acc = 0.0; int cnt = 0;
    for (int ty = 0; ty < tys; ++ty) {
        for (int tx = 0; tx < txs; ++tx) {
            const int i = ty * txs + tx;
            const float centerE = entropy[i];
            float sum = 0.0f; int n = 0;
            const int nx[4] = {tx-1, tx+1, tx,   tx};
            const int ny[4] = {ty,   ty,   ty-1, ty+1};
            for (int k = 0; k < 4; ++k) {
                if (nx[k] < 0 || ny[k] < 0 || nx[k] >= txs || ny[k] >= tys) continue;
                sum += std::fabs(entropy[ny[k]*txs + nx[k]] - centerE);
                ++n;
            }
            if (n > 0) { acc += (sum/n); ++cnt; }
        }
    }
    return (cnt > 0) ? static_cast<float>(acc / cnt) : 0.0f;
}

ZoomResult evaluateZoomTarget(const std::vector<float>& entropy,
                              const std::vector<float>& contrast,
                              int tilesX, int tilesY,
                              int width, int height,
                              float2 currentOffset, float zoom,
                              float2 previousOffset,
                              ZoomState& /*state*/) noexcept
{
    ZoomResult out{};
    const int total = (tilesX > 0 && tilesY > 0) ? (tilesX * tilesY) : 0;
    const bool haveEntropy  = (total > 0) && ((int)entropy.size()  >= total);
    const bool haveContrast = (total > 0) && ((int)contrast.size() >= total);

    auto applyStepWorld = [&](float dirx, float diry, float stepNdc) {
        double wx, wy; ndcToWorldScales(width, height, zoom, wx, wy);
        const double dx = (double)dirx * (double)stepNdc * wx;
        const double dy = (double)diry * (double)stepNdc * wy;
        const double nx = (double)previousOffset.x + dx;
        const double ny = (double)previousOffset.y + dy;
        out.newOffsetX = (float)((1.0 - kBLEND_A) * (double)previousOffset.x + kBLEND_A * nx);
        out.newOffsetY = (float)((1.0 - kBLEND_A) * (double)previousOffset.y + kBLEND_A * ny);
        const float ddx = out.newOffsetX - previousOffset.x;
        const float ddy = out.newOffsetY - previousOffset.y;
        out.distance = std::sqrt(ddx*ddx + ddy*ddy);
        out.shouldZoom = true;
    };

    // --- Kein Signal → deterministischer Drift (aus Innenbereich raus) ---
    if (!haveEntropy || !haveContrast) {
        if (!(Settings::ForceAlwaysZoom)) return out;

        float dirx = g_dirInit ? g_prevDirX : 1.0f;
        float diry = g_dirInit ? g_prevDirY : 0.0f;

        bool inBulb = insideCardioidOrBulb(currentOffset.x, currentOffset.y);
        if (inBulb) antiVoidDriftNDC(currentOffset.x, currentOffset.y, dirx, diry);
        if (!normalize2D(dirx, diry)) { dirx = 1.0f; diry = 0.0f; }

        rotateTowardsLimited(g_prevDirX, g_prevDirY, dirx, diry, kTURN_MAX_RAD);
        dirx = g_prevDirX; diry = g_prevDirY; g_dirInit = true;

        const float stepNdc = clampf(inBulb ? (kSEED_STEP_NDC * 4.0f) : kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);
        applyStepWorld(dirx, diry, stepNdc);

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][DRIFT] stepNdc=%.4f dir=(%.3f,%.3f) d=%.6f",
                           (double)stepNdc, (double)dirx, (double)diry, (double)out.distance);
        }
        return out;
    }

    // --- Robuste Bewertung via Median/MAD ---
    std::vector<float> e = entropy;
    std::vector<float> c = contrast;

    const float eMed = median_inplace(e);
    const float eMAD = mad_from_center_inplace(e, eMed);
    const float cMed = median_inplace(c);
    const float cMAD = mad_from_center_inplace(c, cMed);

    int   bestI = -1;
    float bestS = -1e30f;

    for (int i = 0; i < total; ++i) {
        const float ze = (entropy[i]  - eMed) / eMAD;
        const float zc = (contrast[i] - cMed) / cMAD;
        const float s  = kALPHA_E * ze + kBETA_C * zc;
        if (s > bestS) { bestS = s; bestI = i; }
    }

    // Schwaches/degeneriertes Signal → Drift mit Boost
    if (bestI < 0 || !(eMAD > 1e-6f) || !(cMAD > 1e-6f) || bestS < kMIN_SCORE_SIGMA) {
        if (!(Settings::ForceAlwaysZoom)) return out;

        float dirx = g_dirInit ? g_prevDirX : 1.0f;
        float diry = g_dirInit ? g_prevDirY : 0.0f;
        bool inBulb = insideCardioidOrBulb(currentOffset.x, currentOffset.y);
        if (inBulb) antiVoidDriftNDC(currentOffset.x, currentOffset.y, dirx, diry);
        if (!normalize2D(dirx, diry)) { dirx = 1.0f; diry = 0.0f; }

        rotateTowardsLimited(g_prevDirX, g_prevDirY, dirx, diry, kTURN_MAX_RAD);
        dirx = g_prevDirX; diry = g_prevDirY; g_dirInit = true;

        const float stepNdc = clampf(inBulb ? (kSEED_STEP_NDC * 4.0f) : kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);
        applyStepWorld(dirx, diry, stepNdc);

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][DRIFT*] bestS=%.3f eMAD=%.3g cMAD=%.3g → drift stepNdc=%.4f",
                           (double)bestS, (double)eMAD, (double)cMAD, (double)stepNdc);
        }
        return out;
    }

    // --- Ziel-Tile → Richtung = NDC-Vektor zum Tilezentrum ---
    out.bestIndex  = bestI;
    out.isNewTarget = true;

    float ndcTX = 0.0f, ndcTY = 0.0f;
    tileIndexToNdcCenter(tilesX, tilesY, bestI, ndcTX, ndcTY);

    float tdx = ndcTX, tdy = ndcTY;
    float hx = g_dirInit ? g_prevDirX : 1.0f;
    float hy = g_dirInit ? g_prevDirY : 0.0f;
    rotateTowardsLimited(hx, hy, tdx, tdy, kTURN_MAX_RAD);

    g_prevDirX = hx; g_prevDirY = hy; g_dirInit = true;

    const float stepNdc = clampf(kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);
    applyStepWorld(hx, hy, stepNdc);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][TARGET] idx=%d bestS=%.3f ndc=(%.3f,%.3f) stepNdc=%.4f d=%.6f",
                       bestI, (double)bestS, (double)ndcTX, (double)ndcTY, (double)stepNdc, (double)out.distance);
    }
    return out;
}

void evaluateAndApply(::FrameContext& fctx,
                      ::RendererState& state,
                      ZoomState& bus,
                      float /*gain*/) noexcept
{
    (void)bus;
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
    fctx.newOffset  = make_float2(zr.newOffsetX, zr.newOffsetY);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][EVAL] tiles=%dx%d should=%d new=(%.9f,%.9f)",
                       tilesX, tilesY, zr.shouldZoom ? 1 : 0,
                       (double)fctx.newOffset.x, (double)fctx.newOffset.y);
    }
}

} // namespace ZoomLogic
