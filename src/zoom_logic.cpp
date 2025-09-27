///// Datei: src/zoom_logic.cpp
// Otter: Silk-Lite auto-pan/zoom controller (lokal, kartengetrieben, ohne Heatmap-Zwang).
// Schneefuchs: Deterministische Schrittweite; MSVC-/WX-fest; keine API-Drifts.
// Maus: Richtung = lokaler Kontrastgradient; wählt nächstgelegene valide Kachel; ASCII-only.

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
constexpr float kSEED_STEP_NDC = 0.015f; // Basisschritt in NDC-Anteil der Halbbreite/-höhe
constexpr float kSTEP_MAX_NDC  = 0.35f;  // Sicherheitskappe
constexpr float kBLEND_A       = 0.22f;  // Low-Pass fürs Offset (sanft)
constexpr int   kSEARCH_R      = 2;      // Suchfenster in Kacheln um Bildmitte (+/-R)

// relative Schwelle (gegen "flat") nur aus Kontrast:
constexpr float kTHR_MED_ADD_MAD = 0.5f; // Schwelle = cMed + 0.5 * cMAD

inline float clampf(float x, float a, float b) { return (x < a) ? a : (x > b ? b : x); }

inline bool normalize2D(float& x, float& y) {
    const float n2 = x*x + y*y;
    if (!(n2 > 0.0f)) return false;
    const float inv = 1.0f/std::sqrt(n2);
    x *= inv; y *= inv; return true;
}

// Hauptmengen Test (Cardioid + Periode-2-Bulb)
inline bool insideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q*(q + xm) < 0.25*y*y) return true;
    const double dx = x + 1.0;
    return (dx*dx + y*y) < 0.0625;
}

// Median/MAD (in-place, robust gegen Ausreißer)
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

inline int idxAt(int x, int y, int tilesX) { return y * tilesX + x; }

// edgeScore: lokale Kanten-Plausibilität nur aus Kontrast
inline float edgeScoreAt(const std::vector<float>& contrast,
                         int x, int y, int tilesX, int tilesY)
{
    const int i = idxAt(x,y,tilesX);
    const float lc = contrast[(size_t)i];

    float sum = 0.f; int n=0;
    const int nx4[4] = {x-1, x+1, x,   x};
    const int ny4[4] = {y,   y,   y-1, y+1};
    for (int k=0;k<4;++k) {
        const int xn = nx4[k], yn = ny4[k];
        if (xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        sum += std::fabs(contrast[(size_t)idxAt(xn,yn,tilesX)] - lc);
        ++n;
    }
    const float neighborDiff = (n>0) ? (sum / (float)n) : 0.f;
    return lc + 0.75f * neighborDiff;
}

} // namespace

namespace ZoomLogic {

ZoomResult evaluateZoomTarget(const std::vector<float>& /*entropy*/,
                              const std::vector<float>& contrast,
                              int tilesX, int tilesY,
                              int /*width*/, int /*height*/,
                              float2 /*currentOffset*/, float zoom,
                              float2 previousOffset,
                              ZoomState& /*state*/) noexcept
{
    ZoomResult out{};

    const int total = (tilesX > 0 && tilesY > 0) ? (tilesX * tilesY) : 0;
    if (total <= 0 || static_cast<int>(contrast.size()) < total) {
        // Kein Signal → optionaler Drift
        out.shouldZoom = Settings::ForceAlwaysZoom;
        return out;
    }

    // Robust-Statistik über Kontrast (global, gering)
    std::vector<float> c = contrast;
    const float cMed = median_inplace(c);
    const float cMAD = mad_from_center_inplace(c, cMed);
    const float thr  = cMed + kTHR_MED_ADD_MAD * cMAD;

    // Center-Kachel
    const int cx = tilesX/2, cy = tilesY/2;

    // Suche im lokalen Fenster (naheliegende Kante bevorzugen)
    int   bestI   = -1;
    float bestS   = -1e30f;
    float bestD2  =  1e30f;
    float bestNdcX=  0.0f, bestNdcY = 0.0f;

    for (int dy = -kSEARCH_R; dy <= kSEARCH_R; ++dy) {
        for (int dx = -kSEARCH_R; dx <= kSEARCH_R; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x<0||y<0||x>=tilesX||y>=tilesY) continue;

            const float s = edgeScoreAt(contrast, x, y, tilesX, tilesY);
            if (!(s > thr)) continue;

            float ndcX=0.f, ndcY=0.f;
            tileIndexToNdcCenter(tilesX, tilesY, idxAt(x,y,tilesX), ndcX, ndcY);
            const float d2 = ndcX*ndcX + ndcY*ndcY;

            // Auswahl: primär minimale Distanz, sekundär höherer Score
            if (d2 < bestD2 - 1e-6f || (std::fabs(d2-bestD2) <= 1e-6f && s > bestS)) {
                bestD2   = d2;
                bestS    = s;
                bestI    = idxAt(x,y,tilesX);
                bestNdcX = ndcX;
                bestNdcY = ndcY;
            }
        }
    }

    if (bestI < 0) {
        // Nichts Überzeugendes in Reichweite → Drift (oder Stop, falls gewünscht)
        out.shouldZoom = Settings::ForceAlwaysZoom;
        return out;
    }

    out.bestIndex   = bestI;
    out.isNewTarget = true;

    // Richtung = lokaler Kontrastgradient an bestI (zeigt auf die Kante)
    const int bx = bestI % tilesX, by = bestI / tilesX;

    auto ndcOf = [&](int X,int Y){ float nx,ny; tileIndexToNdcCenter(tilesX, tilesY, idxAt(X,Y,tilesX), nx, ny); return make_float2(nx,ny); };
    const float2 pC = ndcOf(bx,by);

    float2 grad = make_float2(0,0);
    const int nx4b[4] = {bx-1,bx+1,bx,  bx};
    const int ny4b[4] = {by,  by,  by-1,by+1};
    for (int k=0;k<4;++k) {
        const int xn = nx4b[k], yn = ny4b[k];
        if (xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
        const float w = contrast[(size_t)idxAt(xn,yn,tilesX)] - contrast[(size_t)bestI];
        const float2 pN = ndcOf(xn,yn);
        float vx = pN.x - pC.x, vy = pN.y - pC.y;
        if (normalize2D(vx,vy)) { grad.x += w*vx; grad.y += w*vy; }
    }

    float dirx, diry;
    if (std::fabs(grad.x)+std::fabs(grad.y) > 0.f) {
        dirx = grad.x; diry = grad.y; // auf Kante zu
        normalize2D(dirx, diry);
    } else {
        // Fallback: direkt zum Kachelzentrum
        dirx = bestNdcX; diry = bestNdcY;
        if (!normalize2D(dirx, diry)) { dirx = 1.f; diry = 0.f; }
    }

    // Grobe Schrittgröße (exakt in evaluateAndApply)
    const float invZ    = 1.0f / std::max(1e-6f, zoom);
    const float baseStep= clampf(kSEED_STEP_NDC, 0.0f, kSTEP_MAX_NDC);

    out.shouldZoom = true;
    out.newOffsetX = previousOffset.x + dirx * (baseStep * invZ);
    out.newOffsetY = previousOffset.y + diry * (baseStep * invZ);
    out.distance   = baseStep * invZ;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM][TARGET] i=%d ndc=(%.3f,%.3f) dir=(%.3f,%.3f) score=%.3f thr=%.3f d2=%.4f",
                       bestI, (double)bestNdcX, (double)bestNdcY, (double)dirx, (double)diry, (double)bestS, (double)thr, (double)bestD2);
    }
    return out;
}

void evaluateAndApply(::FrameContext& fctx,
                      ::RendererState& state,
                      ZoomState& /*bus*/,
                      float /*gain*/) noexcept
{
    if (CudaInterop::getPauseZoom()) {
        fctx.shouldZoom = false;
        fctx.newOffset  = fctx.offset;
        fctx.newOffsetD = { (double)fctx.offset.x, (double)fctx.offset.y };
        return;
    }

    // Einheitliches Raster für Zielwahl (Overlay-Grid)
    const int overlayTilePx = std::max(1,
        (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));
    const int tilesX = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
    const int tilesY = (fctx.height + overlayTilePx - 1) / overlayTilePx;

    const float2 prevF = fctx.offset;

    ZoomResult zr = evaluateZoomTarget(
        /*entropy*/  state.h_entropy, // ungeachtet; nur Signatur
        /*contrast*/ state.h_contrast,
        tilesX, tilesY,
        fctx.width, fctx.height,
        fctx.offset, fctx.zoom,
        prevF,
        /*state*/ *(ZoomState*)nullptr // ungenutzt; nur Signatur
    );

    if (!zr.shouldZoom) {
        fctx.shouldZoom = false;
        fctx.newOffset  = prevF;
        fctx.newOffsetD = { (double)prevF.x, (double)prevF.y };
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][EVAL] should=0 keep=(%.9f,%.9f)",
                           (double)fctx.newOffset.x, (double)fctx.newOffset.y);
        }
        return;
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

    // Richtung aus zr ableiten (Differenz prev -> proposed)
    double dirx = (double)(zr.newOffsetX - prevF.x);
    double diry = (double)(zr.newOffsetY - prevF.y);
    {
        const double n2 = dirx*dirx + diry*diry;
        if (!(n2 > 0.0)) { dirx = 1.0; diry = 0.0; }
        else { const double inv = 1.0/std::sqrt(n2); dirx *= inv; diry *= inv; }
    }

    // Schrittgröße in Weltkoordinaten
    const double halfW = 0.5 * (double)fctx.width  * std::fabs(stepX);
    const double halfH = 0.5 * (double)fctx.height * std::fabs(stepY);

    // Basis NDC-Schritt (leicht adaptiv: minimaler Fix-Anteil + Distanz zum Zentrum der Zielkachel)
    float ndcTX=0.f, ndcTY=0.f;
    if (zr.bestIndex >= 0) tileIndexToNdcCenter(tilesX, tilesY, zr.bestIndex, ndcTX, ndcTY);
    const float distNdc = std::sqrt(ndcTX*ndcTX + ndcTY*ndcTY);
    const double stepNdc = (double)clampf(kSEED_STEP_NDC * (0.6f + 0.8f * distNdc), 0.6f * kSEED_STEP_NDC, kSTEP_MAX_NDC);

    double moveX = dirx * (stepNdc * halfW);
    double moveY = diry * (stepNdc * halfH);

    // Min. 1/2 Pixel-Schritt sichern
    const double minPixStep = std::min(std::fabs(stepX), std::fabs(stepY));
    const double movLen = std::hypot(moveX, moveY);
    if (movLen < 0.5 * minPixStep && movLen > 0.0) {
        const double s = (0.5 * minPixStep) / movLen;
        moveX *= s; moveY *= s;
    } else if (!(movLen > 0.0)) {
        moveX = minPixStep; moveY = 0.0;
    }

    // In-Set-Veto: Zielzentrum der Kachel (Welt) prüfen → auf Nachbar mit höchstem edgeScore umschalten
    if (zr.bestIndex >= 0) {
        const int bx = zr.bestIndex % tilesX, by = zr.bestIndex / tilesX;

        const double px = (double)ndcTX * 0.5 * (double)fctx.width;
        const double py = (double)ndcTY * 0.5 * (double)fctx.height;
        const double wx = (double)state.center.x + px * stepX;
        const double wy = (double)state.center.y + py * stepY;

        if (insideCardioidOrBulb(wx, wy)) {
            // Nachbar mit max edgeScore wählen (falls verfügbar)
            int bestN = -1; float bestS = -1e30f; float ndcNX=0.f, ndcNY=0.f;
            const int nx4[4] = {bx-1, bx+1, bx,   bx};
            const int ny4[4] = {by,   by,   by-1, by+1};
            for (int k=0;k<4;++k) {
                const int xn = nx4[k], yn = ny4[k];
                if (xn<0||yn<0||xn>=tilesX||yn>=tilesY) continue;
                const float s = edgeScoreAt(state.h_contrast, xn, yn, tilesX, tilesY);
                if (s > bestS) {
                    bestS = s; bestN = idxAt(xn,yn,tilesX);
                    tileIndexToNdcCenter(tilesX, tilesY, bestN, ndcNX, ndcNY);
                }
            }
            if (bestN >= 0) {
                double dx = (double)ndcNX, dy = (double)ndcNY;
                const double n2 = dx*dx + dy*dy;
                if (n2 > 0.0) { const double inv = 1.0/std::sqrt(n2); dx*=inv; dy*=inv; }
                moveX = dx * (stepNdc * halfW);
                moveY = dy * (stepNdc * halfH);
            }
        }
    }

    // Blend auf Basis des aktuellen (double-)Centers
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
        LUCHS_LOG_HOST("[ZOOM][APPLY] i=%d ndc=(%.3f,%.3f) move=(%.3e,%.3e) minPx=%.3e ulp=(%.3e,%.3e) new=(%.9f,%.9f)",
                       zr.bestIndex, (double)ndcTX, (double)ndcTY, moveX, moveY, minPixStep, ulpX, ulpY, txD, tyD);
    }
}

} // namespace ZoomLogic
