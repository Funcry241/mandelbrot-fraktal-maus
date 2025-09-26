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
#include "capybara_mapping.cuh" // capy_pixel_steps_from_zoom_scale(...)

#include <algorithm>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

namespace {

// ----------------------------- Tunables ---------------------------------
constexpr float kSEED_STEP_NDC = 0.015f; // Basisschritt in NDC (Anteil der Bild-Halbdimension)
constexpr float kSTEP_MAX_NDC  = 0.35f;  // Sicherheitskappe
constexpr float kTURN_MAX_RAD  = 0.18f;  // max. Richtungsänderung/Frame
constexpr float kBLEND_A       = 0.22f;  // Low-Pass fürs Offset

// Heatmap-Auswertung
constexpr float kALPHA_E      = 1.0f;  // Gewicht Entropy
constexpr float kBETA_C       = 0.5f;  // Gewicht Contrast
constexpr float kGRAD_BOOST   = 0.60f; // Verstärkung lokaler Nachbarschaftsdifferenz
constexpr float kEDGE_PENALTY = 0.25f; // Randkappung (NDC-Radius^2)
constexpr float kACCEPT_MIN   = 0.35f; // Mindestscore – sonst Drift-Fallback

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
    return (mad > 1e-6f) ? mad : 1.0f;
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
    (void)width; (void)height;
    ZoomResult out{};

    const int total = (tilesX > 0 && tilesY > 0) ? (tilesX * tilesY) : 0;
    const bool haveEntropy  = (total > 0) && (static_cast<int>(entropy.size())  >= total);
    const bool haveContrast = (total > 0) && (static_cast<int>(contrast.size()) >= total);

    // Kein Signal → deterministischer Drift
    if (!haveEntropy || !haveContrast) {
        out.shouldZoom = Settings::ForceAlwaysZoom;
        if (!out.shouldZoom) return out;

        float dirx = g_dirInit ? g_prevDirX : 1.0f;
        float diry = g_dirInit ? g_prevDirY : 0.0f;

        if (insideCardioidOrBulb(currentOffset.x, currentOffset.y)) {
            antiVoidDriftNDC(currentOffset.x, currentOffset.y, dirx, diry);
        }
        if (!normalize2D(dirx, diry)) { dirx = 1.0f; diry = 0.0f; }

        rotateTowardsLimited(g_prevDirX, g_prevDirY, dirx, diry, kTURN_MAX_RAD);
        dirx = g_prevDirX; diry = g_prevDirY;

        const float invZ = 1.0f / std::max(1e-6f, zoom);
        const float step = clampf(kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);
        out.newOffsetX = previousOffset.x + dirx * (step * invZ);
        out.newOffsetY = previousOffset.y + diry * (step * invZ);

        const float dx = out.newOffsetX - previousOffset.x;
        const float dy = out.newOffsetY - previousOffset.y;
        out.distance = std::sqrt(dx*dx + dy*dy);
        out.shouldZoom = true;

        g_dirInit = true;

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][DRIFT] invZ=%.6f stepNdc=%.4f dir=(%.3f,%.3f) d=%.6f",
                           (double)invZ, (double)step, (double)dirx, (double)diry, (double)out.distance);
        }
        return out;
    }

    // Robuste Bewertung via Median/MAD + lokaler Gradientenbonus + Randkappung
    std::vector<float> e = entropy;
    std::vector<float> c = contrast;

    const float eMed = median_inplace(e);
    const float eMAD = mad_from_center_inplace(e, eMed);
    const float cMed = median_inplace(c);
    const float cMAD = mad_from_center_inplace(c, cMed);

    int   bestI = -1;
    float bestS = -1e30f;

    for (int i = 0; i < total; ++i) {
        const int tx = (tilesX > 0) ? (i % tilesX) : 0;
        const int ty = (tilesX > 0) ? (i / tilesX) : 0;

        // Z-Scores
        const float ze = (entropy[i]  - eMed) / (eMAD > 0.0f ? eMAD : 1.0f);
        const float zc = (contrast[i] - cMed) / (cMAD > 0.0f ? cMAD : 1.0f);
        float scoreZ   = kALPHA_E * ze + kBETA_C * zc;

        // Lokale Nachbarschaft (normiert über MAD) → Gradientenbonus
        float sumDiff = 0.0f; int n = 0;
        const int nx4[4] = {tx-1, tx+1, tx,   tx};
        const int ny4[4] = {ty,   ty,   ty-1, ty+1};
        for (int k = 0; k < 4; ++k) {
            const int nx = nx4[k], ny = ny4[k];
            if (nx < 0 || ny < 0 || nx >= tilesX || ny >= tilesY) continue;
            const int j = ny * tilesX + nx;
            const float de = std::fabs(entropy[j]  - entropy[i])  / (eMAD > 0.0f ? eMAD : 1.0f);
            const float dc = std::fabs(contrast[j] - contrast[i]) / (cMAD > 0.0f ? cMAD : 1.0f);
            sumDiff += 0.5f * (de + dc);
            ++n;
        }
        const float localGrad = (n > 0) ? (sumDiff / (float)n) : 0.0f;
        scoreZ *= (1.0f + kGRAD_BOOST * localGrad);

        // Randkappung in NDC
        float ndcX = 0.0f, ndcY = 0.0f;
        tileIndexToNdcCenter(tilesX, tilesY, i, ndcX, ndcY);
        const float r2 = ndcX*ndcX + ndcY*ndcY;
        const float edgeFactor = clampf(1.0f - kEDGE_PENALTY * r2, 0.5f, 1.0f);

        const float s = scoreZ * edgeFactor;

        if (s > bestS) { bestS = s; bestI = i; }
    }

    if (bestI < 0 || !(bestS > kACCEPT_MIN)) {
        // Heatmap vorhanden, aber nichts überzeugend → deterministischer Drift
        out.shouldZoom = Settings::ForceAlwaysZoom;
        return out;
    }

    out.bestIndex  = bestI;
    out.isNewTarget = true;

    // Schrittvorschlag (nur grob; exakte Größe in evaluateAndApply)
    float ndcTX = 0.0f, ndcTY = 0.0f;
    tileIndexToNdcCenter(tilesX, tilesY, bestI, ndcTX, ndcTY);

    // Richtungsglättung gegen vorherige Drift
    float hx = g_dirInit ? g_prevDirX : 1.0f;
    float hy = g_dirInit ? g_prevDirY : 0.0f;
    rotateTowardsLimited(hx, hy, ndcTX, ndcTY, kTURN_MAX_RAD);
    g_prevDirX = hx; g_prevDirY = hy; g_dirInit = true;

    const float invZ = 1.0f / std::max(1e-6f, zoom);
    const float baseStep = clampf(kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);

    out.shouldZoom = true;
    out.newOffsetX = previousOffset.x + hx * (baseStep * invZ);
    out.newOffsetY = previousOffset.y + hy * (baseStep * invZ);

    const float dx = out.newOffsetX - previousOffset.x;
    const float dy = out.newOffsetY - previousOffset.y;
    out.distance = std::sqrt(dx*dx + dy*dy);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][TARGET] idx=%d score=%.3f ndc=(%.3f,%.3f) stepNdc=%.4f invZ=%.6f d=%.6f",
                       bestI, (double)bestS, (double)ndcTX, (double)ndcTY, (double)baseStep, (double)invZ, (double)out.distance);
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
        fctx.newOffsetD = { (double)fctx.offset.x, (double)fctx.offset.y };
        return;
    }

    // --- Einheitliches Raster für Zielwahl: Heatmap-Overlay statt Compute-Grid ---
    const int overlayTilePx = std::max(1,
        (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
    const int tilesY = (fctx.height + overlayTilePx - 1) / overlayTilePx;

    const float2 prevF = fctx.offset;

    ZoomResult zr = evaluateZoomTarget(
        state.h_entropy, state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,   // float: nur für Logging/Heuristik
        prevF,
        bus
    );

    if (zr.shouldZoom) {
        // Zielrichtung ggf. erneut exakt gegen Kachelzentrum ausrichten
        float dirx = g_prevDirX;
        float diry = g_prevDirY;
        if (!normalize2D(dirx, diry)) { dirx = 1.0f; diry = 0.0f; }

        float ndcTX = 0.0f, ndcTY = 0.0f;
        if (zr.bestIndex >= 0) {
            tileIndexToNdcCenter(tilesX, tilesY, zr.bestIndex, ndcTX, ndcTY);
            rotateTowardsLimited(dirx, diry, ndcTX, ndcTY, kTURN_MAX_RAD);
            g_prevDirX = dirx; g_prevDirY = diry; g_dirInit = true;
        }

        // --- Double-präzises Mapping der Schrittweite (NDC → Welt) ---
        double stepX = 0.0, stepY = 0.0;
        capy_pixel_steps_from_zoom_scale(
            (double)state.pixelScale.x,
            (double)state.pixelScale.y,
            fctx.width,
            fctx.zoomD > 0.0 ? fctx.zoomD : (double)state.zoom,
            stepX, stepY
        );

        const double halfW = 0.5 * (double)fctx.width  * std::fabs(stepX);
        const double halfH = 0.5 * (double)fctx.height * std::fabs(stepY);

        // Schrittgröße abhängig von der NDC-Entfernung zum Ziel (fern -> größer)
        const float distNdc = std::sqrt(ndcTX*ndcTX + ndcTY*ndcTY);
        const double stepNdc =
            (double)clampf(kSEED_STEP_NDC * (0.6f + 0.8f * distNdc), 0.6f * kSEED_STEP_NDC, kSTEP_MAX_NDC);

        double moveX = (double)dirx * (stepNdc * halfW);
        double moveY = (double)diry * (stepNdc * halfH);

        // Min. 1/2 Pixel-Schritt sichern
        const double minPixStep = std::min(std::fabs(stepX), std::fabs(stepY));
        const double movLen = std::hypot(moveX, moveY);
        if (movLen < 0.5 * minPixStep && movLen > 0.0) {
            const double s = (0.5 * minPixStep) / movLen;
            moveX *= s; moveY *= s;
        } else if (!(movLen > 0.0)) {
            moveX = minPixStep * (dirx >= 0.0f ? 1.0 : -1.0);
            moveY = 0.0;
        }

        const double baseX = state.center.x;
        const double baseY = state.center.y;

        const double txD = baseX * (1.0 - (double)kBLEND_A) + (baseX + moveX) * (double)kBLEND_A;
        const double tyD = baseY * (1.0 - (double)kBLEND_A) + (baseY + moveY) * (double)kBLEND_A;

        fctx.shouldZoom = true;
        fctx.newOffsetD = { txD, tyD };
        fctx.newOffset  = make_float2((float)txD, (float)tyD);

        if constexpr (Settings::debugLogging) {
            const double ulpX = std::nextafter(baseX, baseX + 1.0) - baseX;
            const double ulpY = std::nextafter(baseY, baseY + 1.0) - baseY;
            LUCHS_LOG_HOST("[ZOOM][APPLY] idx=%d ndc=(%.3f,%.3f) stepNdc=%.4f move=(%.3e,%.3e) minPx=%.3e ulp=(%.3e,%.3e) new=(%.9f,%.9f)",
                           zr.bestIndex, (double)ndcTX, (double)ndcTY, stepNdc, moveX, moveY, minPixStep, ulpX, ulpY, txD, tyD);
        }
    } else {
        fctx.shouldZoom = false;
        fctx.newOffset  = prevF;
        fctx.newOffsetD = { (double)prevF.x, (double)prevF.y };
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][EVAL] tiles=%dx%d should=0 new=(%.9f,%.9f)",
                           tilesX, tilesY, (double)fctx.newOffset.x, (double)fctx.newOffset.y);
        }
    }
}

} // namespace ZoomLogic
