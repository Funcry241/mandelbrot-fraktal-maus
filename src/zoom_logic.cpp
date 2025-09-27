///// Datei: src/zoom_logic.cpp
// Otter: Silk-Lite auto-pan/zoom controller with robust drift fallback (no heatmap required).
// Schneefuchs: Keine API-Drifts; deterministische Schrittweite; MSVC-/WX-fest.
// Maus: Sanfter Drift in NDC, Limitierung der Kurvenrate; ASCII-only.

#include "zoom_logic.hpp"
#include "frame_context.hpp"    // FrameContext
#include "renderer_state.hpp"   // RendererState
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

// Heatmap-Auswertung (schlank, aber robust)
constexpr float kALPHA_E    = 1.0f;  // Gewicht "entropy" (Boundary/Signal)
constexpr float kBETA_C     = 0.5f;  // Gewicht "contrast" (Stddev)
constexpr float kGRAD_BOOST = 0.60f; // Bonus für lokale Kanten
constexpr float kZ_CLAMP    = 6.0f;  // harte Klammer für Z-Scores
constexpr float kBEST_MINZ  = 2.0f;  // bestScore muss >= 2 MAD über Median liegen

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

// Mandelbrot-Hauptmengen-Test (Cardioid + Periode-2-Bulb)
inline bool insideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q*(q + xm) < 0.25*y*y) return true;
    const double dx = x + 1.0;
    return (dx*dx + y*y) < 0.0625;
}

// deterministischer Auswärts-Drift (falls kein Signal)
inline void antiVoidDriftNDC(float& outx, float& outy) {
    outx = 1.0f; outy = 0.0f; normalize2D(outx, outy);
}

// robuste Median/MAD (in-place, schlank)
float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t m = v.size()/2;
    std::nth_element(v.begin(), v.begin()+m, v.end());
    float med = v[m];
    if ((v.size() & 1u) == 0u) {
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

// Richtungs-Gedächtnis (pro Thread)
thread_local bool  g_dirInit  = false;
thread_local float g_prevDirX = 1.0f;
thread_local float g_prevDirY = 0.0f;

} // namespace

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>& entropy,
                              const std::vector<float>& contrast,
                              int tilesX, int tilesY,
                              int /*width*/, int /*height*/,
                              float2 currentOffset, float zoom,
                              float2 previousOffset,
                              ZoomState& /*state*/) noexcept
{
    ZoomResult out{};

    const int total = (tilesX > 0 && tilesY > 0) ? (tilesX * tilesY) : 0;
    const bool haveEntropy  = (total > 0) && ((int)entropy.size()  >= total);
    const bool haveContrast = (total > 0) && ((int)contrast.size() >= total);

    // Kein Signal → deterministischer Drift (in Bildschirmkoord.)
    if (!haveEntropy || !haveContrast) {
        out.shouldZoom = Settings::ForceAlwaysZoom;
        if (!out.shouldZoom) return out;

        float dirx = g_dirInit ? g_prevDirX : 1.0f;
        float diry = g_dirInit ? g_prevDirY : 0.0f;

        if (insideCardioidOrBulb(currentOffset.x, currentOffset.y)) {
            antiVoidDriftNDC(dirx, diry);
        }
        if (!normalize2D(dirx, diry)) { dirx = 1.0f; diry = 0.0f; }
        rotateTowardsLimited(g_prevDirX, g_prevDirY, dirx, diry, kTURN_MAX_RAD);
        dirx = g_prevDirX; diry = g_prevDirY;

        const float invZ = 1.0f / std::max(1e-6f, zoom);
        const float step = clampf(kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);
        out.newOffsetX = previousOffset.x + dirx * (step * invZ);
        out.newOffsetY = previousOffset.y + diry * (step * invZ);
        out.distance   = std::hypot(out.newOffsetX - previousOffset.x,
                                    out.newOffsetY - previousOffset.y);
        out.shouldZoom = true;
        g_dirInit = true;

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][DRIFT] invZ=%.6f stepNdc=%.4f dir=(%.3f,%.3f) d=%.6f",
                           (double)invZ, (double)step, (double)dirx, (double)diry, (double)out.distance);
        }
        return out;
    }

    // --- Robuste Bewertung: Z-Scores (geklemmt) + lokaler Gradientenbonus ---
    std::vector<float> e = entropy, c = contrast;
    const float eMed = median_inplace(e);
    const float eMAD = mad_from_center_inplace(e, eMed);
    const float cMed = median_inplace(c);
    const float cMAD = mad_from_center_inplace(c, cMed);

    std::vector<float> scores((size_t)total);
    for (int i = 0; i < total; ++i) {
        const int tx = (i % std::max(1, tilesX));
        const int ty = (i / std::max(1, tilesX));

        float ze = (entropy[i]  - eMed) / (eMAD > 0.0f ? eMAD : 1.0f);
        float zc = (contrast[i] - cMed) / (cMAD > 0.0f ? cMAD : 1.0f);
        ze = clampf(ze, -kZ_CLAMP, kZ_CLAMP);
        zc = clampf(zc, -kZ_CLAMP, kZ_CLAMP);
        float score = kALPHA_E * ze + kBETA_C * zc;

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
        score *= (1.0f + kGRAD_BOOST * clampf(localGrad, 0.0f, 2.0f)); // geklemmt, stabil

        // Sanfte Randdämpfung (Zentrum bevorzugen, aber nicht hart sperren)
        float nx = 0.0f, ny = 0.0f;
        tileIndexToNdcCenter(tilesX, tilesY, i, nx, ny);
        const float r2 = nx*nx + ny*ny;                       // 0..~2
        const float edge = std::pow(clampf(1.0f - 0.5f*r2, 0.0f, 1.0f), 2.0f);
        scores[(size_t)i] = score * edge;
    }

    // Relative Akzeptanz: bester Score muss deutlich über Median liegen
    std::vector<float> tmp = scores;
    const float sMed = median_inplace(tmp);
    const float sMAD = mad_from_center_inplace(tmp, sMed);

    int   bestI = -1;
    float bestS = -1e30f;
    for (int i = 0; i < total; ++i) {
        const float s = scores[(size_t)i];
        if (s > bestS) { bestS = s; bestI = i; }
    }

    const float bestZ = (bestS - sMed) / (sMAD > 0.0f ? sMAD : 1.0f);
    if (bestI < 0 || !(bestZ >= kBEST_MINZ)) {
        out.shouldZoom = Settings::ForceAlwaysZoom; // Drift-Fallback
        return out;
    }

    // Richtungsupdate zu Kachelzentrum (mit Kurvenlimit)
    float ndcTX = 0.0f, ndcTY = 0.0f;
    tileIndexToNdcCenter(tilesX, tilesY, bestI, ndcTX, ndcTY);

    float hx = g_dirInit ? g_prevDirX : 1.0f;
    float hy = g_dirInit ? g_prevDirY : 0.0f;
    rotateTowardsLimited(hx, hy, ndcTX, ndcTY, kTURN_MAX_RAD);
    g_prevDirX = hx; g_prevDirY = hy; g_dirInit = true;

    const float invZ = 1.0f / std::max(1e-6f, zoom);
    const float step = clampf(kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);

    out.bestIndex  = bestI;
    out.isNewTarget = true;
    out.shouldZoom  = true;
    out.newOffsetX  = previousOffset.x + hx * (step * invZ);
    out.newOffsetY  = previousOffset.y + hy * (step * invZ);
    out.distance    = std::hypot(out.newOffsetX - previousOffset.x,
                                 out.newOffsetY - previousOffset.y);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][TARGET] idx=%d bestZ=%.3f ndc=(%.3f,%.3f) stepNdc=%.4f invZ=%.6f d=%.6f",
                       bestI, (double)bestZ, (double)ndcTX, (double)ndcTY, (double)step, (double)invZ, (double)out.distance);
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

    // Analyse-Raster (screen-constant oder compute-tiling)
    const int overlayTilePx = std::max(1,
        (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
    const int tilesY = (fctx.height + overlayTilePx - 1) / overlayTilePx;

    const float2 prevF = fctx.offset;

    ZoomResult zr = evaluateZoomTarget(
        state.h_entropy, state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,   // float nur für Heuristik
        prevF,
        bus
    );

    if (!zr.shouldZoom) {
        fctx.shouldZoom = false;
        fctx.newOffset  = prevF;
        fctx.newOffsetD = { (double)prevF.x, (double)prevF.y };
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][EVAL] tiles=%dx%d should=0 new=(%.9f,%.9f)",
                           tilesX, tilesY, (double)fctx.newOffset.x, (double)fctx.newOffset.y);
        }
        return;
    }

    // Richtung ggf. nachführen (gegen bestes Kachelzentrum)
    float dirx = g_prevDirX, diry = g_prevDirY;
    if (!normalize2D(dirx, diry)) { dirx = 1.0f; diry = 0.0f; }
    float ndcTX = 0.0f, ndcTY = 0.0f;
    if (zr.bestIndex >= 0) {
        tileIndexToNdcCenter(tilesX, tilesY, zr.bestIndex, ndcTX, ndcTY);
        rotateTowardsLimited(dirx, diry, ndcTX, ndcTY, kTURN_MAX_RAD);
        g_prevDirX = dirx; g_prevDirY = diry; g_dirInit = true;
    }

    // Double-präzise Schrittweite (NDC → Welt)
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

    const float distNdc = std::sqrt(ndcTX*ndcTX + ndcTY*ndcTY);
    const double stepNdc =
        (double)clampf(kSEED_STEP_NDC * (0.6f + 0.8f * distNdc), 0.6f * kSEED_STEP_NDC, kSTEP_MAX_NDC);

    double moveX = (double)dirx * (stepNdc * halfW);
    double moveY = (double)diry * (stepNdc * halfH);

    // min. 1/2 Pixelbewegung
    const double minPixStep = std::min(std::fabs(stepX), std::fabs(stepY));
    const double movLen = std::hypot(moveX, moveY);
    if (movLen < 0.5 * minPixStep && movLen > 0.0) {
        const double s = (0.5 * minPixStep) / movLen;
        moveX *= s; moveY *= s;
    } else if (!(movLen > 0.0)) {
        moveX = minPixStep * (dirx >= 0.0f ? 1.0 : -1.0);
        moveY = 0.0;
    }

    // In-Set-Veto: Kachelzentrum in Welt prüfen und 90° ausweichen
    {
        const double px = (double)ndcTX * 0.5 * (double)fctx.width;
        const double py = (double)ndcTY * 0.5 * (double)fctx.height;
        const double wx = (double)state.center.x + px * stepX;
        const double wy = (double)state.center.y + py * stepY;
        if (insideCardioidOrBulb(wx, wy)) {
            const double tmp = moveX; moveX = -moveY; moveY = tmp;
        }
    }

    // Blend auf Basis des aktuellen double-Centers
    const double baseX = (double)state.center.x;
    const double baseY = (double)state.center.y;

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
}

} // namespace ZoomLogic
