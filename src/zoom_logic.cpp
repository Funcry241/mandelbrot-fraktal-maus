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
static constexpr double kNO_TURN_WARMUP_SEC = 0.3;

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
//      • Begrenze maximale Drehrate (rad/s) → in rad/Frame via dt.
//      • Dämpfe die Translationslänge bei großen Drehwinkeln → keine „seitwärts“-Rucks.
static constexpr float kTURN_OMEGA_MIN = 2.5f;  // rad/s  (sanfte Grunddrehrate)
static constexpr float kTURN_OMEGA_MAX = 10.0f; // rad/s  (volle Drehrate bei starkem Signal/Move)
static constexpr float kTURN_SIG_REF   = 1.00f; // Z-Score-Referenz (stdS) für „volles“ Drehen
static constexpr float kTURN_DIST_REF  = 0.25f; // NDC-Referenzbewegung für „volles“ Drehen
static constexpr float kTHETA_DAMP_LO  = 0.35f; // rad (~20°) Beginn der Dämpfung
static constexpr float kTHETA_DAMP_HI  = 1.20f; // rad (~69°) volle Dämpfung

// ── NEU: Anti-Black-Guard (Warm-up/Niedrigsignal) ----------------------------------------------
static constexpr float kWARMUP_DRIFT_NDC = 0.001f; // NDC-Schritt im Warm-up, weg von Innenbereichen
static constexpr float kVOID_BIAS_NDC    = 0.02f; // NDC-Anteil, der vom Cardioid-Zentrum wegdrückt

// --- robuste Statistik (Median/MAD) ---
static inline float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n   = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if ((n & 1) == 0) {
        // Schneefuchs: korrektes Even-N-Handling via zweitem nth_element
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
        const float m2 = v[mid - 1];
        m = 0.5f * (m + m2);
    }
    return m;
}

// NEU: MAD in-place auf wiederverwendetem Buffer; keine zusätzlichen Allokationen. (Schneefuchs)
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

// NEU: 2D-Normalisierung (robust gegen sehr kleine Beträge). (Schneefuchs)
static inline bool normalize2D(float& x, float& y) {
    const float n2 = x*x + y*y;
    if (n2 <= 1e-20f) return false;
    const float inv = 1.0f / std::sqrt(n2);
    x *= inv; y *= inv;
    return true;
}

// NEU: limitiere Drehung von (dirX,dirY) in Richtung (tx,ty) auf maxAngle (rad). (Otter)
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

// NEU: Analytischer Innen-Test (Cardioid + 2er-Bulb) – harter Ausschluss. (Otter/Schneefuchs)
static inline bool isInsideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q * (q + xm) < 0.25 * y * y) return true; // Hauptkardioide
    const double dx = x + 1.0;                    // Period-2 Bulb (r=0.25)
    if (dx*dx + y*y < 0.0625) return true;
    return false;
}

// NEU: Anti-Black Drift-Vektor berechnen (weg vom Cardioid-/Bulb-Zentrum). (Otter/Schneefuchs)
static inline void computeAntiVoidDriftNDC(float cx, float cy, float& ndcX, float& ndcY) {
    // Richtung weg von (0.25, 0) und zusätzlich weg von (-1, 0) (Period-2 Bulb)
    float vx1 = cx - 0.25f, vy1 = cy - 0.0f;
    float vx2 = cx + 1.0f,  vy2 = cy - 0.0f;
    // Kombiniere Richtungen (gewichtete Summe), normalisiere
    float vx = vx1 + 0.6f * vx2;
    float vy = vy1 + 0.6f * vy2;
    if (!normalize2D(vx, vy)) { vx = 1.0f; vy = 0.0f; }
    ndcX = vx; ndcY = vy;
}

// NEU: NDC-Zentren-Cache pro Geometrie (tilesX, tilesY, width, height). (Schneefuchs)
namespace {
struct NdcCenterCache {
    int tilesX = -1, tilesY = -1, width = -1, height = -1;
    std::vector<float> ndcX;
    std::vector<float> ndcY;

    void ensure(int tx, int ty, int w, int h) {
        const int total = tx * ty;
        if (tx == tilesX && ty == tilesY && w == width && h == height &&
            (int)ndcX.size() == total && (int)ndcY.size() == total) {
            return; // Schneefuchs: Cache hit
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

// thread_local, um Re-Allokationen und false sharing zu vermeiden. (Schneefuchs)
thread_local NdcCenterCache g_ndcCache;

// thread_local Buffer – Allokationen vermeiden. (Schneefuchs)
thread_local std::vector<float> g_bufE;
thread_local std::vector<float> g_bufC;
thread_local std::vector<float> g_bufS;

// Stats-Cache & Cadence (alle kStatsEvery Frames neu) (Schneefuchs)
thread_local int   g_statsTick = 0;
thread_local int   g_statsN    = -1;
thread_local float g_eMed = 0.0f, g_eMad = 1.0f;
thread_local float g_cMed = 0.0f, g_cMad = 1.0f;

// Hysterese/Lock & Retarget – funktion-lokaler Zustand (kein Header-Touch)
static int s_lockLeft = 0;
static int s_sinceRetarget = 0;

// NEU: Vorherige Bewegungsrichtung für Turn-Limiter (persistiert pro Thread). (Otter/Schneefuchs)
thread_local bool  g_dirInit = false;
thread_local float g_prevDirX = 1.0f;
thread_local float g_prevDirY = 0.0f;

// NEU: Vorherige Signalstärke (für dyn. Retarget-Gating). (Schneefuchs)
thread_local float g_prevStdS = 0.0f;
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

    // Mittelwert der lokalen Kontraste (4-Nachbarn)
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

    // Zeitdifferenz dt (sek) für zeitstabile Limits, ohne Header-Änderung. (Schneefuchs)
    static clock::time_point s_lastCall;
    static bool s_haveLast = false;
    double dt = (s_haveLast) ? std::chrono::duration<double>(t0 - s_lastCall).count() : (1.0 / 60.0);
    s_lastCall = t0;
    s_haveLast = true;
    dt = std::max(1.0/240.0, std::min(1.0/15.0, dt)); // clamp gegen Hänger/Spikes

    // ── Warm-up-Timer: ab erstem Aufruf läuft die Uhr. ───────────────────────
    static bool warmupInit = false;
    static clock::time_point warmupStart;
    if (!warmupInit) { warmupStart = t0; warmupInit = true; }
    const double warmupSec = std::chrono::duration<double>(t0 - warmupStart).count();
    const bool freezeDirection = (warmupSec < kNO_TURN_WARMUP_SEC);

    ZoomResult out{};
    out.bestIndex   = -1;
    out.shouldZoom  = false;
    out.isNewTarget = false;
    out.newOffset   = previousOffset; // default: keine Änderung
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

    // ── Anti-Black Warm-up: Falls Center innen → Drift weg vom Innenbereich statt blind zu zoomen.
    if (freezeDirection) {
        if (isInsideCardioidOrBulb((double)currentOffset.x, (double)currentOffset.y)) {
            float nX = 1.0f, nY = 0.0f; computeAntiVoidDriftNDC(currentOffset.x, currentOffset.y, nX, nY);
            const float2 drifted = make_float2(
                previousOffset.x + nX * (kWARMUP_DRIFT_NDC / std::max(1e-6f, zoom)),
                previousOffset.y + nY * (kWARMUP_DRIFT_NDC / std::max(1e-6f, zoom))
            );
            out.newOffset  = drifted;
            out.distance   = std::sqrt((drifted.x-previousOffset.x)*(drifted.x-previousOffset.x) +
                                       (drifted.y-previousOffset.y)*(drifted.y-previousOffset.y));
            out.shouldZoom = true;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3][WARMUP][VOID-GUARD] center_inside -> drift NDC=%.3f (off=(%.5f,%.5f))",
                               kWARMUP_DRIFT_NDC, drifted.x, drifted.y);
            }
        } else {
            out.shouldZoom = true; // normal warm-up, Richtung bleibt
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3][WARMUP] freeze-direction t=%.2fs (limit=%.2fs)",
                               warmupSec, kNO_TURN_WARMUP_SEC);
            }
        }
        return out;
    }

    // Frühzeitiges dyn. Retarget-Throttling (AlwaysZoom) basierend auf vorheriger stdS. (Schneefuchs)
    if (Settings::ForceAlwaysZoom) {
        const float sPrev = clampf(g_prevStdS, 0.0f, 1.0f);
        int dynIntervalPre = static_cast<int>(std::round(
            kRetargetIntervalMax - (kRetargetIntervalMax - kRetargetIntervalMin) * sPrev)); // 8..3
        dynIntervalPre = std::max(kRetargetIntervalMin, std::min(kRetargetIntervalMax, dynIntervalPre));

        if (++s_sinceRetarget < dynIntervalPre) {
            // Zusätzlich: wenn Center innen ist, kleine Void-Bias auch ohne Retarget anwenden
            if (isInsideCardioidOrBulb((double)currentOffset.x, (double)currentOffset.y)) {
                float nX = 1.0f, nY = 0.0f; computeAntiVoidDriftNDC(currentOffset.x, currentOffset.y, nX, nY);
                const float2 drifted = make_float2(
                    previousOffset.x + nX * (kVOID_BIAS_NDC / std::max(1e-6f, zoom)),
                    previousOffset.y + nY * (kVOID_BIAS_NDC / std::max(1e-6f, zoom))
                );
                out.newOffset  = drifted;
                out.distance   = std::sqrt((drifted.x-previousOffset.x)*(drifted.x-previousOffset.x) +
                                           (drifted.y-previousOffset.y)*(drifted.y-previousOffset.y));
            } else {
                out.newOffset = previousOffset;
            }
            out.shouldZoom = true;           // Zoom läuft, Richtung bleibt
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] skip_retarget %d/%d (dyn-pre sPrev=%.3f)",
                               s_sinceRetarget, dynIntervalPre, sPrev);
            }
            return out;
        }
        s_sinceRetarget = 0; // jetzt neu evaluieren
    }

    // Konsistente Länge ableiten
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

    // NDC-Zentren cachen (Schneefuchs)
    g_ndcCache.ensure(tilesX, tilesY, width, height);

    // Robuste Statistik: nur alle kStatsEvery Frames oder wenn N wechselt (Schneefuchs)
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

    // 1. Pass: Scores & Summen
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

    // bestScore/bestIdx (seriell – billig)
    float bestScore = -1e9f;
    int   bestIdx   = -1;
    for (int i = 0; i < N; ++i) {
        const float si = g_bufS[i];
        if (si > bestScore) { bestScore = si; bestIdx = i; }
    }

    // Signalstärke via Z-Score-Streuung
    (void)bestIdx;
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
    const double invZoom   = 1.0 / (double)zoom;
    const float  invTempF  = 1.0f / std::max(1e-6f, temp);

    // ── 2. Pass (FUSIONIERT): Softmax-Reduktion + bestAdj in EINER Schleife ─────────
    double sumW = 0.0, numX = 0.0, numY = 0.0;
    int    interiorSkipped = 0;
    float  bestAdjScore = -1e9f;
    int    bestAdjIdx   = -1;

#ifdef ZOOMLOGIC_OMP
#pragma omp parallel
    {
        float  threadBestScore = -1e9f;
        int    threadBestIdx   = -1;

        // Reduktionsklauseln für die Summen; bestAdj via thread-lokal + critical
#pragma omp for reduction(+:sumW,numX,numY,interiorSkipped) schedule(static)
        for (int i = 0; i < N; ++i) {
            const float si = g_bufS[i];
            if (si < sCutScore) continue;

            // Complex coords des Tile-Zentrums (Offset/Zoom)
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

    // Ergebnis des fusionierten Passes konsumieren
    if (bestAdjIdx >= 0) {
        out.bestIndex    = bestAdjIdx;
        out.bestScore    = bestAdjScore;
        out.bestEntropy  = entropy[bestAdjIdx];
        out.bestContrast = contrast[bestAdjIdx];
    } else {
        // Fallback: alles gefiltert -> initial best
        out.bestIndex    = bestIdx;
        out.bestScore    = bestScore;
        out.bestEntropy  = (bestIdx >= 0) ? entropy[bestIdx]  : 0.0f;
        out.bestContrast = (bestIdx >= 0) ? contrast[bestIdx] : 0.0f;
    }

    // Update prevStdS (EMA) nach aktueller Messung für frühes Gating im nächsten Frame.
    g_prevStdS = 0.85f * g_prevStdS + 0.15f * static_cast<float>(stdS);

    double ndcX = 0.0, ndcY = 0.0;
    if (sumW > 0.0) {
        const double inv = 1.0 / sumW;
        ndcX = numX * inv;
        ndcY = numY * inv;
    }

    // Relative Hysterese + Lock auf Zielwechsel (vor Bewegung)
    if (state.lastAcceptedIndex >= 0 && out.bestIndex >= 0 && out.bestIndex != state.lastAcceptedIndex) {
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

    // ── Bewegungsvektor berechnen (proposed ohne Glättung) ──────────────────────────
    const float2 proposedOffset_raw = make_float2(
        currentOffset.x + (float)(ndcX * invZoom),
        currentOffset.y + (float)(ndcY * invZoom)
    );
    float mvx = proposedOffset_raw.x - previousOffset.x;
    float mvy = proposedOffset_raw.y - previousOffset.y;
    const float rawDist = std::sqrt(mvx*mvx + mvy*mvy);

    // ── NEU: Kleine Void-Bias auch NACH Retarget, falls Center aktuell innen ist
    if (isInsideCardioidOrBulb((double)currentOffset.x, (double)currentOffset.y)) {
        float nX = 1.0f, nY = 0.0f; computeAntiVoidDriftNDC(currentOffset.x, currentOffset.y, nX, nY);
        mvx += nX * (kVOID_BIAS_NDC / std::max(1e-6f, zoom));
        mvy += nY * (kVOID_BIAS_NDC / std::max(1e-6f, zoom));
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3][VOID-GUARD] center_inside -> add bias ndc=%.3f", kVOID_BIAS_NDC);
        }
    }

    // ── NEU: Richtungswechsel "smoother" via zeitstabilem Turn-Limiter + Längendämpfung. (Otter/Schneefuchs)
    float sigFactor  = clampf((float)stdS / kTURN_SIG_REF, 0.0f, 1.0f);
    float distFactor = clampf(rawDist / kTURN_DIST_REF,     0.0f, 1.0f);
    const float omegaMin = kTURN_OMEGA_MIN;
    const float omegaMax = kTURN_OMEGA_MAX;
    const float omega    = omegaMin + (omegaMax - omegaMin) * std::max(sigFactor, distFactor);
    float turnMaxRad     = omega * static_cast<float>(dt); // rad pro Frame

    if (!g_dirInit) {
        // Erste Initialisierung: Richtung aus aktuellem Move ableiten.
        g_prevDirX = (rawDist > 0.0f) ? (mvx / rawDist) : 1.0f;
        g_prevDirY = (rawDist > 0.0f) ? (mvy / rawDist) : 0.0f;
        g_dirInit  = true;
    }

    // Zielrichtung = Richtung des aktuellen Bewegungsvektors (falls vorhanden)
    float tgtDirX = mvx, tgtDirY = mvy;
    const bool hasMove = normalize2D(tgtDirX, tgtDirY);

    // Bei sehr kleinem Bewegungsvektor vermeiden wir Sprünge; halten Richtung.
    if (hasMove) {
        float dirX = g_prevDirX, dirY = g_prevDirY;

        // Zielwinkel vor der Rotation (für Längendämpfung)
        const float preDot = clampf(dirX*tgtDirX + dirY*tgtDirY, -1.0f, 1.0f);
        const float preAng = std::acos(preDot);

        rotateTowardsLimited(dirX, dirY, tgtDirX, tgtDirY, turnMaxRad);

        // Länge dämpfen je nach Drehwinkel (S-Kurve) — nimmt die Härte aus Abbiegungen.
        const float lenScale = 1.0f - smoothstepf(kTHETA_DAMP_LO, kTHETA_DAMP_HI, preAng);

        mvx = dirX * (rawDist * lenScale);
        mvy = dirY * (rawDist * lenScale);

        g_prevDirX = dirX; g_prevDirY = dirY;

        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3] turn(dt=%.3f) preAng=%.3f rad max=%.3f rad lenScale=%.3f dir=(%.3f,%.3f) tgt=(%.3f,%.3f)",
                           dt, preAng, turnMaxRad, lenScale, g_prevDirX, g_prevDirY, tgtDirX, tgtDirY);
        }
    }

    const float2 proposedOffset = make_float2(
        previousOffset.x + mvx,
        previousOffset.y + mvy
    );

    const float dx = proposedOffset.x - previousOffset.x;
    const float dy = proposedOffset.y - previousOffset.y;
    const float dist = std::sqrt(dx*dx + dy*dy);

    // Bewegung glätten (EMA, dt-basiert; τ abhängig von Distanz)
    const float distNorm = clampf(dist / 0.5f, 0.0f, 1.0f);
    const float tau = kEMA_TAU_MAX + (kEMA_TAU_MIN - kEMA_TAU_MAX) * distNorm; // lerp
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

    // Kompatibilitäts-State
    const bool indexChanged = (out.bestIndex != state.lastAcceptedIndex);
    out.isNewTarget = indexChanged && hasSignal;
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
            "[ZOOMV3] move: dist=%.4f emaAlpha=%.3f ndc=(%.4f,%.4f) offRaw=(%.5f,%.5f) newOff=(%.5f,%.5f) tiles=(%d,%d) ms=%.3f",
            dist, emaAlpha, (float)ndcX, (float)ndcY,
            proposedOffset_raw.x, proposedOffset_raw.y,
            out.newOffset.x, out.newOffset.y,
            tilesX, tilesY, ms
        );
        if (interiorSkipped > 0) {
            LUCHS_LOG_HOST("[ZOOMV3] interior_skip=%d/%d (cardioid/2-bulb hard filter)", interiorSkipped, N);
        }
    }

    return out;
}

} // namespace ZoomLogic
