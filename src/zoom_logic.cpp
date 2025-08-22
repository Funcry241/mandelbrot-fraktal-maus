///// src/zoom_logic.cpp
// LUCHS1
// Perf V3 hot path – no header/API change (ASCII logs only)
// Otter: Warm-up ohne Richtungswechsel: erst zoomen, dann lenken. (Bezug zu Otter)
// Schneefuchs: Minimalinvasiv, keine Header-/API-Änderung. ASCII-Logs. (Bezug zu Schneefuchs)
// NEU: Anti-Black-Guard — vermeidet „Zoom ins Schwarze“ (Cardioid/Bulb) in Warm-up & bei schwachem Signal.
//      • Warm-up-Drift: leichter, zielgerichteter Schub aus Innenbereichen heraus (dt-sicher)
//      • Void-Bias: kleiner Auslenkungsanteil weg von der Cardioid-Mitte, falls Center innen ist
//      • Zeitstabil: Yaw-Rate-Limiter in rad/s (dt -> rad/Frame), plus Längendämpfung bei großen Drehwinkeln
//      • dt-basierte EMA mit Zeitkonstante τ(dist): stabil bei Framedrops und Extrembewegung
//      • Turn-Transition Gate + Acc-Limiter: weiche Rampen nach Richtungswechsel (kein harter Antritt)

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef ZOOMLOGIC_OMP
  #include <omp.h>
#endif

// ------- V3-Parameter (lokal, bewusst nicht in settings.hpp, um Hotfixe zu erleichtern) -------
// Scoring-Gewichte
static constexpr float kALPHA_E = 1.00f; // Gewicht Entropie
static constexpr float kBETA_C  = 0.50f; // Gewicht Kontrast
// Softmax
static constexpr float kTEMP_BASE = 1.00f; // Grundtemperatur für Softmax (wird adaptiv skaliert)
// Bewegung / Glättung (EMA: alpha = 1 - exp(-dt/τ))
static constexpr float kEMA_ALPHA_MIN = 0.06f; // untere Klammer
static constexpr float kEMA_ALPHA_MAX = 0.30f; // obere Klammer
static constexpr float kEMA_TAU_MIN   = 0.040f; // s, starke Bewegung → kurze τ → größere alpha
static constexpr float kEMA_TAU_MAX   = 0.220f; // s, feine Bewegung → lange τ → kleine alpha
// Signaldetektion
static constexpr float kMIN_SIGNAL_Z  = 0.15f; // minimale Z-Score-Stärke für "aktives" Signal
// Deadzone (nur Dokumentation – Ausgabe in out.minDistance)
static constexpr float kMIN_DISTANCE  = 0.02f; // ~NDC/Zoom-Skala
// Otter: sanfter Drift auch ohne starkes Signal, wenn AlwaysZoom aktiv ist
static constexpr float kFORCE_MIN_DRIFT_ALPHA = 0.05f;

// NEU: Warm-up-Zeit (Sekunden), in der KEIN Richtungswechsel erfolgt. (Bezug zu Otter)
static constexpr double kNO_TURN_WARMUP_SEC = 1.0;

// NEU: Softmax-Sparsification – ignoriere Beiträge mit sehr kleiner Gewichtung. (Bezug zu Schneefuchs)
static constexpr float kSOFTMAX_LOG_EPS = -7.0f;

// NEU: Relative Hysterese + kurzer Lock gegen Flip-Flop (massive Richtungswechsel) (Otter/Schneefuchs)
static constexpr float kHYST_REL    = 0.12f;
static constexpr int   kLOCK_FRAMES = 12;

// NEU: Retarget-Throttling – dynamisch (CPU-schonend, ruhiger) (Otter/Schneefuchs)
static constexpr int   kRetargetIntervalMin = 3;
static constexpr int   kRetargetIntervalMax = 8;

// NEU: Statistik nur alle M Frames (Kopier-/nth_element-Last reduzieren). (Schneefuchs)
static constexpr int   kStatsEvery = 3;

// ── NEU (Otter/Schneefuchs): Zeitstabiler Richtungswechsel-Limiter (Yaw-Rate in rad/s) + Längendämpfung.
static constexpr float kTURN_OMEGA_MIN = 2.5f;  // rad/s
static constexpr float kTURN_OMEGA_MAX = 10.0f; // rad/s
static constexpr float kTURN_SIG_REF   = 1.00f; // stdS
static constexpr float kTURN_DIST_REF  = 0.25f; // NDC
static constexpr float kTHETA_DAMP_LO  = 0.35f; // rad (~20°)
static constexpr float kTHETA_DAMP_HI  = 1.20f; // rad (~69°)

// ── NEU: Anti-Black-Guard (Warm-up/Niedrigsignal) ----------------------------------------------
static constexpr float kWARMUP_DRIFT_NDC = 0.030f; // sanfter
static constexpr float kVOID_BIAS_NDC    = 0.020f; // zarter Bias

// ── NEU: Turn-Transition Gate (weiche Rampen nach Richtungswechsel) -----------------------------
static constexpr float kTURN_GATE_TRIG   = 0.60f; // rad (~34°)
static constexpr float kTURN_GATE_T_MIN  = 0.18f; // s
static constexpr float kTURN_GATE_T_MAX  = 0.40f; // s
static constexpr float kTURN_GATE_START  = 0.18f; // Start-Skalierung
static constexpr float kSTEP_CAP_NDC     = 0.06f; // max NDC-Step direkt nach Trigger

// ── NEU: Acceleration Limiter -------------------------------------------------------------------
static constexpr float kACC_BASE = 1.50f; // units/s^2; eff. accMax = kACC_BASE / zoom

// --- robuste Statistik (Median/MAD) ---
static inline float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n   = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if ((n & 1) == 0) {
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
        const float m2 = v[mid - 1];
        m = 0.5f * (m + m2);
    }
    return m;
}

static inline float mad_inplace_from_center(std::vector<float>& buf, float med) {
    if (buf.empty()) return 1.0f;
    for (float& x : buf) x = std::fabs(x - med);
    float m = median_inplace(buf);
    return (m > 1e-6f) ? m : 1.0f;
}

// --- kleine Helfer ---
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline float smoothstepf(float a, float b, float x) {
    const float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
static inline float sCurve01(float u) {
    u = clampf(u, 0.0f, 1.0f);
    return u * u * (3.0f - 2.0f * u);
}

// Normalisieren
static inline bool normalize2D(float& x, float& y) {
    const float n2 = x*x + y*y;
    if (n2 <= 1e-20f) return false;
    const float inv = 1.0f / std::sqrt(n2);
    x *= inv; y *= inv;
    return true;
}

// Limitierte Rotation
static inline void rotateTowardsLimited(float& dirX, float& dirY, float tx, float ty, float maxAngle) {
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

// Innen-Test
static inline bool isInsideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q * (q + xm) < 0.25 * y * y) return true; // Hauptkardioide
    const double dx = x + 1.0;                    // Period-2 Bulb (r=0.25)
    if (dx*dx + y*y < 0.0625) return true;
    return false;
}

// Anti-Black Drift-Richtung
static inline void computeAntiVoidDriftNDC(float cx, float cy, float& ndcX, float& ndcY) {
    float vx1 = cx - 0.25f, vy1 = cy - 0.0f;
    float vx2 = cx + 1.0f,  vy2 = cy - 0.0f;
    float vx = vx1 + 0.6f * vx2;
    float vy = vy1 + 0.6f * vy2;
    if (!normalize2D(vx, vy)) { vx = 1.0f; vy = 0.0f; }
    ndcX = vx; ndcY = vy;
}

// ── Caches & Zustände ───────────────────────────────────────────────────────
namespace {
struct NdcCenterCache {
    int tilesX = -1, tilesY = -1, width = -1, height = -1;
    std::vector<float> ndcX;
    std::vector<float> ndcY;

    void ensure(int tx, int ty, int w, int h) {
        const int total = tx * ty;
        if (tx == tilesX && ty == tilesY && w == width && h == height &&
            (int)ndcX.size() == total && (int)ndcY.size() == total) {
            return;
        }
        tilesX = tx; tilesY = ty; width = w; height = h;
        ndcX.resize(total);
        ndcY.resize(total);

        for (int i = 0; i < total; ++i) {
            auto center = tileIndexToPixelCenter(i, tilesX, tilesY, width, height);
            const double cx = static_cast<double>(center.first)  / static_cast<double>(width);
            const double cy = static_cast<double>(center.second) / static_cast<double>(height);
            ndcX[i] = static_cast<float>((cx - 0.5) * 2.0);
            ndcY[i] = static_cast<float>((cy - 0.5) * 2.0);
        }
    }
};

thread_local NdcCenterCache g_ndcCache;

thread_local std::vector<float> g_bufE;
thread_local std::vector<float> g_bufC;
thread_local std::vector<float> g_bufS;

thread_local int   g_statsTick = 0;
thread_local int   g_statsN    = -1;
thread_local float g_eMed = 0.0f, g_eMad = 1.0f;
thread_local float g_cMed = 0.0f, g_cMad = 1.0f;

static int s_lockLeft = 0;
static int s_sinceRetarget = 0;

thread_local bool  g_dirInit = false;
thread_local float g_prevDirX = 1.0f;
thread_local float g_prevDirY = 0.0f;

thread_local float g_prevStdS = 0.0f;

// Turn-Transition-Gate & Acc-Limiter
thread_local bool  g_turnGateActive = false;
thread_local float g_turnGateT      = 0.0f;
thread_local float g_turnGateDur    = 0.0f;
thread_local float g_speed          = 0.0f; // units/s
} // namespace

namespace ZoomLogic {

float computeEntropyContrast(
    const std::vector<float>& entropy,
    int width, int height, int tileSize) noexcept
{
    if (width <= 0 || height <= 0 || tileSize <= 0) return 0.0f;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int total  = tilesX * tilesY;
    if (total <= 0 || (int)entropy.size() < total) return 0.0f;

    double acc = 0.0;
    int cnt = 0;
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            const int i = ty * tilesX + tx;
            const float center = entropy[i];
            float sum = 0.0f; int n = 0;
            const int nx[4] = { tx-1, tx+1, tx,   tx };
            const int ny[4] = { ty,   ty,   ty-1, ty+1 };
            for (int k = 0; k < 4; ++k) {
                if (nx[k] < 0 || ny[k] < 0 || nx[k] >= tilesX || ny[k] >= tilesY) continue;
                const int j = ny[k] * tilesX + nx[k];
                sum += std::fabs(entropy[j] - center);
                ++n;
            }
            if (n > 0) { acc += (sum / n); ++cnt; }
        }
    }
    return (cnt > 0) ? static_cast<float>(acc / cnt) : 0.0f;
}

// V3: Kontinuierlicher Schwerpunkt (Softmax) + EMA-Glättung + No-Black + Hysterese/Lock
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

    // dt
    static clock::time_point s_lastCall;
    static bool s_haveLast = false;
    double dt = (s_haveLast) ? std::chrono::duration<double>(t0 - s_lastCall).count() : (1.0 / 60.0);
    s_lastCall = t0;
    s_haveLast = true;
    dt = std::max(1.0/240.0, std::min(1.0/15.0, dt)); // clamp

    // invZoom EINMAL definiert (robust)
    const double invZoom = 1.0 / std::max(1e-6, (double)zoom);

    // Warm-up-Timer
    static bool warmupInit = false;
    static clock::time_point warmupStart;
    if (!warmupInit) { warmupStart = t0; warmupInit = true; }
    const double warmupSec = std::chrono::duration<double>(t0 - warmupStart).count();
    const bool freezeDirection = (warmupSec < kNO_TURN_WARMUP_SEC);

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
    if (tilesX <= 0 || tilesY <= 0 || totalTiles <= 0) {
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3] invalid tiles: tilesX=%d tilesY=%d", tilesX, tilesY);
        }
        if (Settings::ForceAlwaysZoom) {
            out.shouldZoom = true;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] forceAlwaysZoom=1 -> shouldZoom=1 (invalid geometry fallback)");
            }
        }
        return out;
    }

    // Warm-up: sanfter, rechts-präferenter Drift aus Innenbereichen
    if (freezeDirection) {
        if (isInsideCardioidOrBulb((double)currentOffset.x, (double)currentOffset.y)) {
            float nX = 1.0f, nY = 0.0f;
            computeAntiVoidDriftNDC(currentOffset.x, currentOffset.y, nX, nY);
            if (currentOffset.x < -0.30f && nX < 0.0f) nX = 0.0f;
            if (nX < 0.0f) nX = std::fabs(nX) * 0.35f;

            const float2 target = make_float2(
                previousOffset.x + nX * (kWARMUP_DRIFT_NDC * (float)invZoom),
                previousOffset.y + nY * (kWARMUP_DRIFT_NDC * (float)invZoom)
            );

            const float alphaWarm = 0.20f;
            out.newOffset = make_float2(
                previousOffset.x * (1.0f - alphaWarm) + target.x * alphaWarm,
                previousOffset.y * (1.0f - alphaWarm) + target.y * alphaWarm
            );

            out.distance   = std::sqrt((out.newOffset.x-previousOffset.x)*(out.newOffset.x-previousOffset.x) +
                                       (out.newOffset.y-previousOffset.y)*(out.newOffset.y-previousOffset.y));
            out.shouldZoom = true;

            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3][WARMUP][VOID-GUARD] cx=%.4f nX=%.3f nY=%.3f -> off=(%.5f,%.5f)",
                               currentOffset.x, nX, nY, out.newOffset.x, out.newOffset.y);
            }
        } else {
            out.shouldZoom = true;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3][WARMUP] freeze-direction t=%.2fs (limit=%.2fs)",
                               warmupSec, kNO_TURN_WARMUP_SEC);
            }
        }
        return out;
    }

    // Dyn Retarget-Throttle (AlwaysZoom)
    if (Settings::ForceAlwaysZoom) {
        const float sPrev = clampf(g_prevStdS, 0.0f, 1.0f);
        int dynIntervalPre = static_cast<int>(std::round(
            kRetargetIntervalMax - (kRetargetIntervalMax - kRetargetIntervalMin) * sPrev));
        dynIntervalPre = std::max(kRetargetIntervalMin, std::min(kRetargetIntervalMax, dynIntervalPre));

        if (++s_sinceRetarget < dynIntervalPre) {
            if (isInsideCardioidOrBulb((double)currentOffset.x, (double)currentOffset.y)) {
                float nX = 1.0f, nY = 0.0f; computeAntiVoidDriftNDC(currentOffset.x, currentOffset.y, nX, nY);
                if (currentOffset.x < -0.30f && nX < 0.0f) nX = 0.0f;
                if (nX < 0.0f) nX = std::fabs(nX) * 0.35f;

                const float2 drifted = make_float2(
                    previousOffset.x + nX * (kVOID_BIAS_NDC * (float)invZoom),
                    previousOffset.y + nY * (kVOID_BIAS_NDC * (float)invZoom)
                );
                out.newOffset  = drifted;
                out.distance   = std::sqrt((drifted.x-previousOffset.x)*(drifted.x-previousOffset.x) +
                                           (drifted.y-previousOffset.y)*(drifted.y-previousOffset.y));
            } else {
                out.newOffset = previousOffset;
            }
            out.shouldZoom = true;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] skip_retarget %d/%d (dyn-pre sPrev=%.3f)",
                               s_sinceRetarget, dynIntervalPre, sPrev);
            }
            return out;
        }
        s_sinceRetarget = 0;
    }

    // Konsistente Länge
    const int N = std::min<int>(totalTiles, std::min<int>((int)entropy.size(), (int)contrast.size()));
    if (N <= 0) {
        if (Settings::debugLogging) LUCHS_LOG_HOST("[ZOOMV3] empty metrics -> no zoom");
        if (Settings::ForceAlwaysZoom) {
            out.shouldZoom = true;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] forceAlwaysZoom=1 -> shouldZoom=1 (empty metrics)");
            }
        }
        return out;
    }

    // NDC-Zentren cachen
    g_ndcCache.ensure(tilesX, tilesY, width, height);

    // Statistik (robust, throttled)
    const bool statsNChanged = (g_statsN != N);
    const bool recomputeStats = statsNChanged || ((++g_statsTick % kStatsEvery) == 1);
    if (recomputeStats) {
        g_bufE.reserve(N); g_bufC.reserve(N);
        g_bufE.assign(entropy.begin(),  entropy.begin()  + N);
        g_bufC.assign(contrast.begin(), contrast.begin() + N);
        g_eMed = median_inplace(g_bufE);
        g_eMad = mad_inplace_from_center(g_bufE, g_eMed);
        g_cMed = median_inplace(g_bufC);
        g_cMad = mad_inplace_from_center(g_bufC, g_cMed);
        g_statsN = N;
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3] stats recompute N=%d eMed=%.4f eMAD=%.4f cMed=%.4f cMAD=%.4f",
                           N, g_eMed, g_eMad, g_cMed, g_cMad);
        }
    }
    const float e_med = g_eMed;
    const float e_mad = (g_eMad > 1e-6f) ? g_eMad : 1.0f;
    const float c_med = g_cMed;
    const float c_mad = (g_cMad > 1e-6f) ? g_cMad : 1.0f;

    // 1. Pass: Scores
    g_bufS.resize(N);
    double sumS  = 0.0;
    double sumS2 = 0.0;

#ifdef ZOOMLOGIC_OMP
#pragma omp parallel for reduction(+:sumS,sumS2) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        const float ez = (entropy[i]  - e_med) / e_mad;
        const float cz = (contrast[i] - c_med) / c_mad;
        const float si = kALPHA_E * ez + kBETA_C * cz;
        g_bufS[i] = si;
        sumS  += (double)si;
        sumS2 += (double)si * (double)si;
    }

    float bestScore = -1e9f;
    int   bestIdx   = -1;
    for (int i = 0; i < N; ++i) {
        const float si = g_bufS[i];
        if (si > bestScore) { bestScore = si; bestIdx = i; }
    }

    const double meanS = sumS / std::max(1, N);
    const double varS  = std::max(0.0, (sumS2 / std::max(1, N)) - meanS * meanS);
    const double stdS  = std::sqrt(varS);
    const bool   hasSignal = (stdS >= kMIN_SIGNAL_Z);

    // Softmax-Temperatur adaptiv
    float temp = kTEMP_BASE;
    if (stdS > 1e-6) temp = static_cast<float>(kTEMP_BASE / (0.5f + (float)stdS));
    temp = clampf(temp, 0.2f, 2.5f);

    // Softmax-Schwelle & Konstanten
    const float  sMax      = bestScore;
    const float  sCutScore = sMax + temp * kSOFTMAX_LOG_EPS;
    const float  invTempF  = 1.0f / std::max(1e-6f, temp);

    // 2. Pass: Softmax-Reduktion + bestAdj
    double sumW = 0.0, numX = 0.0, numY = 0.0;
    int    interiorSkipped = 0;
    float  bestAdjScore = -1e9f;
    int    bestAdjIdx   = -1;

#ifdef ZOOMLOGIC_OMP
#pragma omp parallel
    {
        float  threadBestScore = -1e9f;
        int    threadBestIdx   = -1;

#pragma omp for reduction(+:sumW,numX,numY,interiorSkipped) schedule(static)
        for (int i = 0; i < N; ++i) {
            const float si = g_bufS[i];
            if (si < sCutScore) continue;

            const double cx = (double)currentOffset.x + (double)g_ndcCache.ndcX[i] * invZoom;
            const double cy = (double)currentOffset.y + (double)g_ndcCache.ndcY[i] * invZoom;

            if (isInsideCardioidOrBulb(cx, cy)) { ++interiorSkipped; continue; }

            const float  sShift = (si - sMax) * invTempF;
            const double w      = std::exp(static_cast<double>(sShift));
            sumW += w;
            numX += w * (double)g_ndcCache.ndcX[i];
            numY += w * (double)g_ndcCache.ndcY[i];

            if (si > threadBestScore) { threadBestScore = si; threadBestIdx = i; }
        }

#pragma omp critical
        {
            if (threadBestScore > bestAdjScore) {
                bestAdjScore = threadBestScore;
                bestAdjIdx   = threadBestIdx;
            }
        }
    }
#else
    for (int i = 0; i < N; ++i) {
        const float si = g_bufS[i];
        if (si < sCutScore) continue;

        const double cx = (double)currentOffset.x + (double)g_ndcCache.ndcX[i] * invZoom;
        const double cy = (double)currentOffset.y + (double)g_ndcCache.ndcY[i] * invZoom;
        if (isInsideCardioidOrBulb(cx, cy)) { ++interiorSkipped; continue; }

        const float  sShift = (si - sMax) * invTempF;
        const double w      = std::exp(static_cast<double>(sShift));
        sumW += w;
        numX += w * (double)g_ndcCache.ndcX[i];
        numY += w * (double)g_ndcCache.ndcY[i];

        if (si > bestAdjScore) { bestAdjScore = si; bestAdjIdx = i; }
    }
#endif

    if (bestAdjIdx >= 0) {
        out.bestIndex    = bestAdjIdx;
        out.bestScore    = bestAdjScore;
        out.bestEntropy  = entropy[bestAdjIdx];
        out.bestContrast = contrast[bestAdjIdx];
    } else {
        out.bestIndex    = bestIdx;
        out.bestScore    = bestScore;
        out.bestEntropy  = (bestIdx >= 0) ? entropy[bestIdx]  : 0.0f;
        out.bestContrast = (bestIdx >= 0) ? contrast[bestIdx] : 0.0f;
    }

    // prevStdS aktualisieren
    g_prevStdS = 0.85f * g_prevStdS + 0.15f * static_cast<float>(stdS);

    // Bewegungsvektor (roh)
    double ndcX = 0.0, ndcY = 0.0;
    if (sumW > 0.0) {
        const double inv = 1.0 / sumW;
        ndcX = numX * inv;
        ndcY = numY * inv;
    }

    const bool indexChanged = (out.bestIndex != state.lastAcceptedIndex);
    if (state.lastAcceptedIndex >= 0 && out.bestIndex >= 0 && indexChanged) {
        if (s_lockLeft > 0) {
            --s_lockLeft;
            out.bestIndex    = state.lastAcceptedIndex;
            out.bestScore    = state.lastAcceptedScore;
            out.bestEntropy  = entropy[out.bestIndex];
            out.bestContrast = contrast[out.bestIndex];
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] lock_hold last=%d left=%d", out.bestIndex, s_lockLeft);
            }
        } else {
            const double rel = (state.lastAcceptedScore > 1e-12f)
                ? (double(out.bestScore) / double(state.lastAcceptedScore)) : 1e9;
            if (rel < (1.0 + kHYST_REL)) {
                out.bestIndex    = state.lastAcceptedIndex;
                out.bestScore    = state.lastAcceptedScore;
                out.bestEntropy  = entropy[out.bestIndex];
                out.bestContrast = contrast[out.bestIndex];
                if (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ZOOMV3] hysteresis_keep last=%d rel=%.3f thr=%.3f",
                                   state.lastAcceptedIndex, rel, 1.0 + kHYST_REL);
                }
            } else {
                s_lockLeft = kLOCK_FRAMES;
                if (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ZOOMV3] switch_accept old=%d new=%d rel=%.3f lock=%d",
                                   state.lastAcceptedIndex, out.bestIndex, rel, s_lockLeft);
                }
            }
        }
    }

    const float2 proposedOffset_raw = make_float2(
        currentOffset.x + (float)(ndcX * invZoom),
        currentOffset.y + (float)(ndcY * invZoom)
    );
    float mvx = proposedOffset_raw.x - previousOffset.x;
    float mvy = proposedOffset_raw.y - previousOffset.y;
    float rawDist = std::sqrt(mvx*mvx + mvy*mvy);

    // Void-Bias nach Retarget (rechts-präferent)
    if (isInsideCardioidOrBulb((double)currentOffset.x, (double)currentOffset.y)) {
        float nX = 1.0f, nY = 0.0f; computeAntiVoidDriftNDC(currentOffset.x, currentOffset.y, nX, nY);
        if (currentOffset.x < -0.30f && nX < 0.0f) nX = 0.0f;
        if (nX < 0.0f) nX = std::fabs(nX) * 0.35f;

        mvx += nX * (kVOID_BIAS_NDC * (float)invZoom);
        mvy += nY * (kVOID_BIAS_NDC * (float)invZoom);
        rawDist = std::sqrt(mvx*mvx + mvy*mvy);
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3][VOID-GUARD] center_inside -> add bias ndc=%.3f (nX=%.3f nY=%.3f)",
                           kVOID_BIAS_NDC, nX, nY);
        }
    }

    // Yaw-Limiter + Längendämpfung
    float preAngForGate = 0.0f; bool havePreAng = false;
    {
        float tgtDirX = mvx, tgtDirY = mvy;
        const bool hasMove = normalize2D(tgtDirX, tgtDirY);
        if (hasMove) {
            float dirX = g_prevDirX, dirY = g_prevDirY;
            const float preDot = clampf(dirX*tgtDirX + dirY*tgtDirY, -1.0f, 1.0f);
            const float preAng = std::acos(preDot);
            havePreAng = true; preAngForGate = preAng;

            const float sigFactor  = clampf((float)stdS / kTURN_SIG_REF, 0.0f, 1.0f);
            const float distFactor = clampf(rawDist   / kTURN_DIST_REF,  0.0f, 1.0f);
            const float omega      = kTURN_OMEGA_MIN + (kTURN_OMEGA_MAX - kTURN_OMEGA_MIN) * std::max(sigFactor, distFactor);
            const float turnMaxRad = omega * static_cast<float>(dt);

            rotateTowardsLimited(dirX, dirY, tgtDirX, tgtDirY, turnMaxRad);

            const float lenScale = 1.0f - smoothstepf(kTHETA_DAMP_LO, kTHETA_DAMP_HI, preAng);
            const float newLen   = rawDist * lenScale;
            mvx = dirX * newLen;
            mvy = dirY * newLen;

            g_prevDirX = dirX; g_prevDirY = dirY;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] turn(dt=%.3f) preAng=%.3f rad max=%.3f rad lenScale=%.3f dir=(%.3f,%.3f)",
                               dt, preAng, turnMaxRad, lenScale, g_prevDirX, g_prevDirY);
            }
        } else if (!g_dirInit && rawDist > 0.0f) {
            g_prevDirX = mvx / rawDist; g_prevDirY = mvy / rawDist; g_dirInit = true;
        }
    }

    // Turn-Transition Gate (sanfter Ramp-Up)
    {
        const bool strongTurn = (havePreAng && preAngForGate > kTURN_GATE_TRIG);
        bool triggerGate = (strongTurn || (indexChanged && hasSignal));
        if (triggerGate && !g_turnGateActive) {
            const float ang01 = havePreAng
                              ? clampf((preAngForGate - kTURN_GATE_TRIG) / (kTHETA_DAMP_HI - kTURN_GATE_TRIG), 0.0f, 1.0f)
                              : 1.0f;
            float dur = kTURN_GATE_T_MIN + (kTURN_GATE_T_MAX - kTURN_GATE_T_MIN) * ang01;
            const float sig01 = clampf((float)stdS / kTURN_SIG_REF, 0.0f, 1.0f);
            dur *= (0.85f + (1.0f - sig01) * 0.30f);
            g_turnGateActive = true;
            g_turnGateT      = 0.0f;
            g_turnGateDur    = dur;
        }

        if (g_turnGateActive) {
            g_turnGateT += static_cast<float>(dt);
            float u = (g_turnGateDur > 1e-6f) ? (g_turnGateT / g_turnGateDur) : 1.0f;
            const float ramp = kTURN_GATE_START + (1.0f - kTURN_GATE_START) * sCurve01(u);

            const float stepCap = kSTEP_CAP_NDC * (float)invZoom;
            const float curLen  = std::sqrt(mvx*mvx + mvy*mvy);
            float targetLen     = curLen * ramp;
            if (g_turnGateT <= 2.0f * (float)dt) targetLen = std::min(targetLen, stepCap);

            if (curLen > 1e-12f) {
                const float s = targetLen / curLen;
                mvx *= s; mvy *= s;
            }

            if (u >= 1.0f) g_turnGateActive = false;
        }
    }

    // Acceleration Limiter (zoomskaliert)
    {
        const float curLen = std::sqrt(mvx*mvx + mvy*mvy);
        const float desiredV = (dt > 1e-6) ? (curLen / (float)dt) : 0.0f;
        const float accMax   = kACC_BASE * (float)invZoom;
        const float dvMax    = accMax * (float)dt;

        if (!std::isfinite(g_speed)) g_speed = 0.0f;
        float dv = desiredV - g_speed;
        if (dv >  dvMax) dv =  dvMax;
        if (dv < -dvMax) dv = -dvMax;
        g_speed += dv;
        const float newLen = g_speed * (float)dt;

        if (curLen > 1e-12f) {
            const float s = (newLen > 0.0f) ? (newLen / curLen) : 0.0f;
            mvx *= s; mvy *= s;
        }
    }

    const float2 proposedOffset = make_float2(
        previousOffset.x + mvx,
        previousOffset.y + mvy
    );

    const float dx = proposedOffset.x - previousOffset.x;
    const float dy = proposedOffset.y - previousOffset.y;
    const float dist = std::sqrt(dx*dx + dy*dy);

    // EMA-Glättung (dt-basiert, τ abhängig von Distanz)
    const float distNorm = clampf(dist / 0.5f, 0.0f, 1.0f);
    const float tau = kEMA_TAU_MAX + (kEMA_TAU_MIN - kEMA_TAU_MAX) * distNorm;
    float emaAlpha = 1.0f - std::exp(-static_cast<float>(dt) / std::max(1e-5f, tau));
    emaAlpha = clampf(emaAlpha, kEMA_ALPHA_MIN, kEMA_ALPHA_MAX);
    if (Settings::ForceAlwaysZoom && !hasSignal) {
        emaAlpha = std::max(emaAlpha, kFORCE_MIN_DRIFT_ALPHA);
    }

    const float2 smoothed = make_float2(
        previousOffset.x * (1.0f - emaAlpha) + proposedOffset.x * emaAlpha,
        previousOffset.y * (1.0f - emaAlpha) + proposedOffset.y * emaAlpha
    );

    // Ausgabe
    out.distance   = dist;
    out.newOffset  = (hasSignal || Settings::ForceAlwaysZoom) ? smoothed : previousOffset;
    out.shouldZoom = (hasSignal || Settings::ForceAlwaysZoom);

    // State-Update
    const bool indexChangedFinal = (out.bestIndex != state.lastAcceptedIndex);
    out.isNewTarget = indexChangedFinal && hasSignal;
    if (hasSignal && out.bestIndex >= 0) {
        const bool first = (state.lastAcceptedIndex < 0);
        state.lastAcceptedIndex = out.bestIndex;
        state.lastAcceptedScore = first ? out.bestScore
                                        : (0.98f * state.lastAcceptedScore + 0.02f * out.bestScore);
        state.lastOffset        = out.newOffset;
        state.lastTilesX        = tilesX;
        state.lastTilesY        = tilesY;
        state.cooldownLeft      = 0;
    }

    // Logging (ASCII only)
    if (Settings::debugLogging) {
        const int bx = (out.bestIndex >= 0 && tilesX > 0) ? out.bestIndex % tilesX : -1;
        const int by = (out.bestIndex >= 0 && tilesX > 0) ? out.bestIndex / tilesX : -1;
        const auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LUCHS_LOG_HOST(
            "[ZOOMV3] best=%d tile=(%d,%d) e=%.4f c=%.4f score=%.3f | stdS=%.3f temp=%.3f signal=%d",
            out.bestIndex, bx, by, out.bestEntropy, out.bestContrast, out.bestScore,
            (float)stdS, temp, hasSignal ? 1 : 0
        );
        LUCHS_LOG_HOST(
            "[ZOOMV3] move: dist=%.4f emaAlpha=%.3f gate=%d gateT=%.3f/%.3f vel=%.4f ms=%.3f off=(%.5f,%.5f)",
            dist, emaAlpha, g_turnGateActive ? 1 : 0, g_turnGateT, g_turnGateDur, g_speed, ms,
            out.newOffset.x, out.newOffset.y
        );
        if (interiorSkipped > 0) {
            LUCHS_LOG_HOST("[ZOOMV3] interior_skip=%d/%d (cardioid/2-bulb hard filter)", interiorSkipped, N);
        }
    }

    return out;
}

} // namespace ZoomLogic
