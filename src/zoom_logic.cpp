///// src/zoom_logic.cpp — V3-lite (Gradient Escape)
// Ziel: Keine "harten" Richtungswechsel + nie blind ins Schwarze.
// - Gradient-basierte Anti-Void-Richtung (Cardioid / 2er-Bulb)
// - Escape-Guard bei Innenlage & schwachem Signal (Pan gewinnt gegen frühen Zoom)
// - Softmax-Zielwahl (robust via Median/MAD), aber schlank
// - Turn-Limiter (zeitstabil) + Längendämpfung bei großen Drehwinkeln
// - dt-EMA (tau(dist)) für glatte Bewegung
//
// Nur ASCII-Logs über LUCHS_LOG_HOST, keine Header-/API-Änderung.

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp" // tileIndexToPixelCenter

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

// ---- Gewichte & Schwellwerte -------------------------------------------------
constexpr float kALPHA_E = 1.00f;    // Entropie-Gewicht
constexpr float kBETA_C  = 0.50f;    // Kontrast-Gewicht
constexpr float kTEMP_BASE = 1.00f;  // Softmax-Basis-Temperatur
constexpr float kSOFTMAX_LOG_EPS = -7.0f; // sehr kleine Beiträge ignorieren
constexpr float kMIN_SIGNAL_STD = 0.15f;  // Minimum-Std der Scores für aktives Signal

// Warm-up / Anti-Black & Seed
constexpr double kNO_TURN_WARMUP_SEC = 1.0;   // erste Sekunde keine Richtungswechsel
constexpr float  kWARMUP_DRIFT_NDC   = 0.08f; // Drift weg vom Innenbereich (Cardioid/Bulb)
constexpr float  kSEED_STEP_NDC      = 0.015f;// winziger Schritt entlang letzter Richtung

// Escape-Guard (nach Warm-up): stärkerer Pan raus aus innen bei schwachem Signal
constexpr float  kESCAPE_STD_THR   = 0.20f;   // strenger als kMIN_SIGNAL_STD
constexpr float  kESCAPE_STEP_NDC  = 0.12f;   // kräftigerer Schritt für verlässlichen Ausstieg

// Turn-Limiter (max. Drehgeschwindigkeit) & Längendämpfung bei großen Drehwinkeln
constexpr float kTURN_OMEGA_MIN = 2.5f;  // rad/s
constexpr float kTURN_OMEGA_MAX = 10.0f; // rad/s
constexpr float kTHETA_DAMP_LO  = 0.35f; // rad ~20°
constexpr float kTHETA_DAMP_HI  = 1.20f; // rad ~69°

// EMA (dt-basiert): alpha = 1 - exp(-dt/τ)
constexpr float kEMA_TAU_MIN   = 0.040f; // s — schnelle Bewegungen
constexpr float kEMA_TAU_MAX   = 0.220f; // s — feine Bewegungen
constexpr float kEMA_ALPHA_MIN = 0.06f;  // Untergrenze
constexpr float kEMA_ALPHA_MAX = 0.30f;  // Obergrenze
constexpr float kFORCE_MIN_DRIFT_ALPHA = 0.05f; // Minimum bei AlwaysZoom & schwachem Signal

// UI/Kompatibilität
constexpr float kMIN_DISTANCE = 0.02f; // NDC/Zoom-Skala (nur Ausgabe)

// ---- Kleine Helfer -----------------------------------------------------------
inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
inline float smoothstepf(float a, float b, float x) {
    const float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
inline bool normalize2D(float& x, float& y) {
    const float n2 = x*x + y*y;
    if (n2 <= 1e-20f) return false;
    const float inv = 1.0f / std::sqrt(n2);
    x *= inv; y *= inv;
    return true;
}
inline void rotateTowardsLimited(float& dirX, float& dirY, float tx, float ty, float maxAngle) {
    if (!normalize2D(tx, ty)) return;
    if (!normalize2D(dirX, dirY)) { dirX = tx; dirY = ty; return; }
    const float dot = clampf(dirX*tx + dirY*ty, -1.0f, 1.0f);
    const float ang = std::acos(dot);
    if (!(ang > 0.0f) || ang <= maxAngle) { dirX = tx; dirY = ty; return; }
    const float crossZ = dirX*ty - dirY*tx;
    const float rot    = (crossZ >= 0.0f) ? maxAngle : -maxAngle;
    const float c = std::cos(rot), s = std::sin(rot);
    const float nx = c*dirX - s*dirY;
    const float ny = s*dirX + c*dirY;
    dirX = nx; dirY = ny;
}
inline bool insideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q * (q + xm) < 0.25 * y * y) return true; // Hauptkardioide
    const double dx = x + 1.0;                    // Period-2 Bulb (r=0.25)
    if (dx*dx + y*y < 0.0625) return true;
    return false;
}

// Gradient-basierte Ausstiegsrichtung (zeigt sicher nach „außen“)
inline void antiVoidOutwardNDC_Gradient(float cx, float cy, float& ndcX, float& ndcY) {
    // Cardioid: f(x,y) = q*(q + xm) - 0.25*y^2,  q=(x-0.25)^2 + y^2
    const double xm = double(cx) - 0.25;
    const double q  = xm*xm + double(cy)*double(cy);
    const double dfx = (q + 4.0*q*xm + 2.0*xm*xm);
    const double dfy = double(cy) * (4.0*q + 2.0*xm - 0.5);

    // Bulb (r=0.25): g(x,y)=(x+1)^2 + y^2 - 0.0625,  ∇g=(2(x+1), 2y)
    const double bx = double(cx) + 1.0;
    const double by = double(cy);
    const double dgx = 2.0*bx;
    const double dgy = 2.0*by;

    // Entscheidung: näher an Cardioid- oder Bulb-Innen?
    // (Wir nutzen die Gradienten-Längen als Heuristik)
    const double nf = std::hypot(dfx, dfy);
    const double ng = std::hypot(dgx, dgy);

    double vx, vy;
    if (nf >= ng) { vx = dfx; vy = dfy; }
    else          { vx = dgx; vy = dgy; }

    float fx = float(vx), fy = float(vy);
    if (!normalize2D(fx, fy)) { fx = 1.0f; fy = 0.0f; }
    ndcX = fx; ndcY = fy;
}

// Robuste Median/MAD (in-place)
float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size(); const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if ((n & 1) == 0) {
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
        m = 0.5f * (m + v[mid - 1]);
    }
    return m;
}
float mad_from_center_inplace(std::vector<float>& v, float med) {
    if (v.empty()) return 1.0f;
    for (float& x : v) x = std::fabs(x - med);
    const float m = median_inplace(v);
    return (m > 1e-6f) ? m : 1.0f;
}

// Persistenter Bewegungszustand (pro Thread)
thread_local bool  g_dirInit  = false;
thread_local float g_prevDirX = 1.0f;
thread_local float g_prevDirY = 0.0f;

} // namespace

namespace ZoomLogic {

// (Optional) Metrik für HUD/Stats
float computeEntropyContrast(
    const std::vector<float>& entropy,
    int width, int height, int tileSize) noexcept
{
    if (width <= 0 || height <= 0 || tileSize <= 0) return 0.0f;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int total  = tilesX * tilesY;
    if ((int)entropy.size() < total) return 0.0f;

    double acc = 0.0; int cnt = 0;
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            const int i = ty * tilesX + tx;
            const float c = entropy[i];
            const int nx[4] = { tx-1, tx+1, tx,   tx };
            const int ny[4] = { ty,   ty,   ty-1, ty+1 };
            float sum = 0.0f; int n = 0;
            for (int k = 0; k < 4; ++k) {
                if (nx[k] < 0 || ny[k] < 0 || nx[k] >= tilesX || ny[k] >= tilesY) continue;
                sum += std::fabs(entropy[ny[k]*tilesX + nx[k]] - c);
                ++n;
            }
            if (n > 0) { acc += (sum / n); ++cnt; }
        }
    }
    return (cnt > 0) ? static_cast<float>(acc / cnt) : 0.0f;
}

// Hauptlogik
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

    // dt (sek) – robust clamp gegen Spikes
    static clock::time_point s_last;
    static bool s_haveLast = false;
    double dt = s_haveLast ? std::chrono::duration<double>(t0 - s_last).count() : (1.0 / 60.0);
    s_last = t0; s_haveLast = true;
    dt = std::clamp(dt, 1.0/240.0, 1.0/15.0);

    // Warm-up-Fenster (keine Richtungswechsel)
    static bool warmInit = false;
    static clock::time_point warmStart;
    if (!warmInit) { warmStart = t0; warmInit = true; }
    const double warmSec = std::chrono::duration<double>(t0 - warmStart).count();
    const bool freezeDirection = (warmSec < kNO_TURN_WARMUP_SEC);

    ZoomResult out{};
    out.bestIndex   = -1;
    out.shouldZoom  = false;
    out.isNewTarget = false;
    out.newOffset   = previousOffset;
    out.minDistance = kMIN_DISTANCE;
    out.bestScore   = 0.0f;
    out.bestEntropy = 0.0f;
    out.bestContrast= 0.0f;
    out.distance    = 0.0f;

    // Geometrie prüfen
    const int totalTiles = tilesX * tilesY;
    if (tilesX <= 0 || tilesY <= 0 || totalTiles <= 0 ||
        (int)entropy.size() < totalTiles || (int)contrast.size() < totalTiles)
    {
        out.shouldZoom = Settings::ForceAlwaysZoom;
        return out;
    }

    const double invZoom = 1.0 / std::max(1e-6f, zoom);

    // --- Warm-up: NIE "in place" ------------------------------------------------
    if (freezeDirection) {
        out.shouldZoom = true;
        if (insideCardioidOrBulb(currentOffset.x, currentOffset.y)) {
            float nx=1.0f, ny=0.0f;
            antiVoidOutwardNDC_Gradient(currentOffset.x, currentOffset.y, nx, ny);
            const float escScale = kWARMUP_DRIFT_NDC * float(invZoom * (1.0 + 0.4 * std::min<double>(zoom, 8.0)));
            const float2 target = make_float2(previousOffset.x + nx * escScale,
                                              previousOffset.y + ny * escScale);
            const float a = 0.20f;
            out.newOffset = make_float2(previousOffset.x*(1.0f-a) + target.x*a,
                                        previousOffset.y*(1.0f-a) + target.y*a);
        } else {
            const float sx = g_dirInit ? g_prevDirX : 1.0f;
            const float sy = g_dirInit ? g_prevDirY : 0.0f;
            const float2 target = make_float2(previousOffset.x + sx * float(kSEED_STEP_NDC*invZoom),
                                              previousOffset.y + sy * float(kSEED_STEP_NDC*invZoom));
            const float a = 0.20f;
            out.newOffset = make_float2(previousOffset.x*(1.0f-a) + target.x*a,
                                        previousOffset.y*(1.0f-a) + target.y*a);
        }
        out.distance = std::hypot(out.newOffset.x - previousOffset.x,
                                  out.newOffset.y - previousOffset.y);
        return out;
    }

    // --- Robuste Scores (Median/MAD) --------------------------------------------
    std::vector<float> e(entropy.begin(),  entropy.begin()  + totalTiles);
    std::vector<float> c(contrast.begin(), contrast.begin() + totalTiles);
    const float eMed = median_inplace(e);
    const float eMad = mad_from_center_inplace(e, eMed);
    const float cMed = median_inplace(c);
    const float cMad = mad_from_center_inplace(c, cMed);

    // Std-Abweichung der Scores (Signalstärke)
    double sumS = 0.0, sumS2 = 0.0;
    float  bestScore = -1e9f; int bestIdx = -1;
    std::vector<float> ndcX(totalTiles), ndcY(totalTiles);

    for (int i = 0; i < totalTiles; ++i) {
        auto p = tileIndexToPixelCenter(i, tilesX, tilesY, width, height);
        const double cx = double(p.first)  / double(width);
        const double cy = double(p.second) / double(height);
        ndcX[i] = float((cx - 0.5) * 2.0);
        ndcY[i] = float((cy - 0.5) * 2.0);

        const float ez = (entropy[i]  - eMed) / (eMad > 1e-6f ? eMad : 1.0f);
        const float cz = (contrast[i] - cMed) / (cMad > 1e-6f ? cMad : 1.0f);
        const float s  = kALPHA_E * ez + kBETA_C * cz;
        sumS  += (double)s;
        sumS2 += (double)s * (double)s;
        if (s > bestScore) { bestScore = s; bestIdx = i; }
    }
    const double meanS = sumS / std::max(1, totalTiles);
    const double varS  = std::max(0.0, (sumS2 / std::max(1, totalTiles)) - meanS * meanS);
    const double stdS  = std::sqrt(varS);
    const bool   hasSignal = (stdS >= kMIN_SIGNAL_STD);

    // --- Escape-Guard: Innen + schwaches Signal → RAUS (Pan gewinnt) ------------
    const bool centerInside = insideCardioidOrBulb(currentOffset.x, currentOffset.y);
    bool usedEscape = false;
    float2 rawTarget = previousOffset; // wird unten gesetzt

    if (centerInside && (stdS < kESCAPE_STD_THR)) {
        float ex=1.0f, ey=0.0f;
        antiVoidOutwardNDC_Gradient(currentOffset.x, currentOffset.y, ex, ey);
        const float escScale = kESCAPE_STEP_NDC * float(invZoom * (1.0 + 0.4 * std::min<double>(zoom, 8.0)));
        rawTarget = make_float2(previousOffset.x + ex * escScale,
                                previousOffset.y + ey * escScale);
        usedEscape = true;
    } else {
        // --- Softmax-Zielwahl (außerhalb Innenbereiche) --------------------------
        float temp = kTEMP_BASE;
        if (stdS > 1e-6) temp = float(kTEMP_BASE / (0.5 + stdS));
        temp = clampf(temp, 0.2f, 2.5f);

        const float sCut     = bestScore + temp * kSOFTMAX_LOG_EPS;
        const float invTemp  = 1.0f / std::max(1e-6f, temp);

        double sumW = 0.0, numX = 0.0, numY = 0.0;
        int bestAdj = -1; float bestAdjScore = -1e9f;

        for (int i = 0; i < totalTiles; ++i) {
            const float ez = (entropy[i]  - eMed) / (eMad > 1e-6f ? eMad : 1.0f);
            const float cz = (contrast[i] - cMed) / (cMad > 1e-6f ? cMad : 1.0f);
            const float s  = kALPHA_E * ez + kBETA_C * cz;
            if (s < sCut) continue;

            const double cx = currentOffset.x + ndcX[i] * invZoom;
            const double cy = currentOffset.y + ndcY[i] * invZoom;
            if (insideCardioidOrBulb(cx, cy)) continue;

            const double w = std::exp(double((s - bestScore) * invTemp));
            sumW += w; numX += w * ndcX[i]; numY += w * ndcY[i];
            if (s > bestAdjScore) { bestAdjScore = s; bestAdj = i; }
        }

        double ndcTX = 0.0, ndcTY = 0.0;
        if (sumW > 0.0) { const double inv = 1.0 / sumW; ndcTX = numX * inv; ndcTY = numY * inv; }
        else if (bestAdj >= 0) { ndcTX = ndcX[bestAdj]; ndcTY = ndcY[bestAdj]; }
        else if (bestIdx >= 0) {
            const double cx = currentOffset.x + ndcX[bestIdx]*invZoom;
            const double cy = currentOffset.y + ndcY[bestIdx]*invZoom;
            if (!insideCardioidOrBulb(cx, cy)) { ndcTX = ndcX[bestIdx]; ndcTY = ndcY[bestIdx]; }
        }

        // Fallback: falls immer noch 0 → Richtung aus Gradient (außen) oder letzte Richtung
        if (ndcTX == 0.0 && ndcTY == 0.0) {
            if (centerInside) {
                float bx=1.0f, by=0.0f; antiVoidOutwardNDC_Gradient(currentOffset.x, currentOffset.y, bx, by);
                ndcTX = bx; ndcTY = by;
            } else {
                ndcTX = g_dirInit ? g_prevDirX : 1.0f;
                ndcTY = g_dirInit ? g_prevDirY : 0.0f;
            }
        }

        rawTarget = make_float2(previousOffset.x + float(ndcTX * invZoom),
                                previousOffset.y + float(ndcTY * invZoom));
        out.bestIndex = (sumW > 0.0) ? 0 : bestIdx; // reine Kompatibilitätsangabe, nicht kritisch
    }

    // --- Bewegungsvektor (roh) --------------------------------------------------
    float mvx = rawTarget.x - previousOffset.x;
    float mvy = rawTarget.y - previousOffset.y;
    const float rawDist = std::sqrt(mvx*mvx + mvy*mvy);

    // --- Richtungswechsel glätten (Turn-Limiter + Längendämpfung) ---------------
    float dirX = g_dirInit ? g_prevDirX : (rawDist > 0.0f ? mvx/rawDist : 1.0f);
    float dirY = g_dirInit ? g_prevDirY : (rawDist > 0.0f ? mvy/rawDist : 0.0f);
    g_dirInit = true;

    float tgtX = mvx, tgtY = mvy;
    const bool hasMove = normalize2D(tgtX, tgtY);

    const float sigFactor  = clampf(float(stdS), 0.0f, 1.0f);
    const float distFactor = clampf(rawDist / 0.25f, 0.0f, 1.0f);
    const float omega      = kTURN_OMEGA_MIN + (kTURN_OMEGA_MAX - kTURN_OMEGA_MIN) * std::max(sigFactor, distFactor);
    const float maxTurn    = omega * float(dt);

    float lenScale = 1.0f;
    if (hasMove) {
        const float preDot = clampf(dirX*tgtX + dirY*tgtY, -1.0f, 1.0f);
        const float preAng = std::acos(preDot);
        rotateTowardsLimited(dirX, dirY, tgtX, tgtY, maxTurn);
        lenScale = 1.0f - smoothstepf(kTHETA_DAMP_LO, kTHETA_DAMP_HI, preAng); // große Abbiegung → kürzerer Schritt
        g_prevDirX = dirX; g_prevDirY = dirY;
    }

    const float2 proposed = make_float2(previousOffset.x + dirX * (rawDist * lenScale),
                                        previousOffset.y + dirY * (rawDist * lenScale));

    // --- EMA (dt-basiert) -------------------------------------------------------
    const float dist = std::hypot(proposed.x - previousOffset.x, proposed.y - previousOffset.y);
    const float distNorm = clampf(dist / 0.5f, 0.0f, 1.0f);
    const float tau = kEMA_TAU_MAX + (kEMA_TAU_MIN - kEMA_TAU_MAX) * distNorm; // lerp
    float emaAlpha = 1.0f - std::exp(-float(dt) / std::max(1e-5f, tau));
    emaAlpha = clampf(emaAlpha, kEMA_ALPHA_MIN, kEMA_ALPHA_MAX);
    if (Settings::ForceAlwaysZoom && !hasSignal) {
        emaAlpha = std::max(emaAlpha, kFORCE_MIN_DRIFT_ALPHA);
    }

    const float2 smoothed = make_float2(
        previousOffset.x * (1.0f - emaAlpha) + proposed.x * emaAlpha,
        previousOffset.y * (1.0f - emaAlpha) + proposed.y * emaAlpha
    );

    // --- Ausgabe ----------------------------------------------------------------
    out.distance   = std::hypot(smoothed.x - previousOffset.x, smoothed.y - previousOffset.y);
    out.newOffset  = smoothed;
    out.shouldZoom = usedEscape || hasSignal || Settings::ForceAlwaysZoom;

    // Minimaler State (keine komplexe Hysterese/Locks nötig)
    out.isNewTarget = (out.bestIndex >= 0 && out.bestIndex != state.lastAcceptedIndex && hasSignal);
    if (out.isNewTarget) state.lastAcceptedIndex = out.bestIndex;
    state.lastOffset = out.newOffset;
    state.lastTilesX = tilesX;
    state.lastTilesY = tilesY;
    state.cooldownLeft = 0;

#if 1
    if (Settings::debugLogging) {
        const auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LUCHS_LOG_HOST("[ZOOM-LITE] escape=%d stdS=%.3f hasSig=%d turnMax=%.3f len=%.3f ema=%.3f dist=%.4f ms=%.3f",
                       usedEscape ? 1 : 0, (float)stdS, hasSignal ? 1 : 0,
                       kTURN_OMEGA_MAX * float(dt), // Obergrenze nur indikativ
                       lenScale, emaAlpha, out.distance, ms);
    }
#endif

    return out;
}

} // namespace ZoomLogic
