// Perf V3 hot path – no header/API change (ASCII logs only)
// Otter: Warm-up ohne Richtungswechsel: erst zoomen, dann lenken. (Bezug zu Otter)
// Schneefuchs: Minimalinvasiv, keine Header-/API-Änderung. ASCII-Logs. (Bezug zu Schneefuchs)

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
// Bewegung / Glättung
static constexpr float kEMA_ALPHA_MIN = 0.06f; // minimale Glättung
static constexpr float kEMA_ALPHA_MAX = 0.30f; // maximale Glättung
// Signaldetektion
static constexpr float kMIN_SIGNAL_Z  = 0.15f; // minimale Z-Score-Stärke für "aktives" Signal
// Deadzone (nur Dokumentation – Ausgabe in out.minDistance)
static constexpr float kMIN_DISTANCE  = 0.02f; // ~NDC/Zoom-Skala
// Otter: sanfter Drift auch ohne starkes Signal, wenn AlwaysZoom aktiv ist
static constexpr float kFORCE_MIN_DRIFT_ALPHA = 0.05f;

// NEU: Warm-up-Zeit (Sekunden), in der KEIN Richtungswechsel erfolgt.
//      Zoom läuft weiter, aber Offset bleibt unverändert. (Bezug zu Otter)
static constexpr double kNO_TURN_WARMUP_SEC = 1.0;

// NEU: Softmax-Sparsification – ignoriere Beiträge mit sehr kleiner Gewichtung.
//      si < sMax + temp * kSOFTMAX_LOG_EPS -> Beitrag vernachlässigbar.
//      exp(-7) ≈ 0.0009, ausreichend klein für Schwerpunkt. (Bezug zu Schneefuchs)
static constexpr float kSOFTMAX_LOG_EPS = -7.0f;

// NEU: Relative Hysterese + kurzer Lock gegen Flip-Flop (massive Richtungswechsel)
//      (Bezug zu Otter/Schneefuchs)
static constexpr float kHYST_REL    = 0.12f; // +12% besser nötig für Zielwechsel
static constexpr int   kLOCK_FRAMES = 12;    // nach Wechsel so viele Frames sperren

// NEU: Retarget-Throttling – nur alle N Frames neu auswerten (CPU-schonend, ruhiger)
//      (Bezug zu Schneefuchs/Otter)
static constexpr int   kRetargetInterval = 5; // nur alle 5 Frames neu zielen (bei ForceAlwaysZoom)

// NEU: Statistik nur alle M Frames (Kopier-/nth_element-Last reduzieren). (Bezug zu Schneefuchs)
static constexpr int   kStatsEvery = 3; // alle 3 Frames Median/MAD neu

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

// NEU: MAD in-place auf wiederverwendetem Buffer; keine zusätzlichen Allokationen.
//      (Bezug zu Schneefuchs)
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

// NEU: Analytischer Innen-Test (Cardioid + 2er-Bulb) – harter Ausschluss.
//      Minimalinvasiv: wir prüfen die komplexen Koordinaten des Kachelzentrums.
//      (Bezug zu Otter/Schneefuchs)
static inline bool isInsideCardioidOrBulb(double x, double y) noexcept {
    const double xm = x - 0.25;
    const double q  = xm*xm + y*y;
    if (q * (q + xm) < 0.25 * y * y) return true; // Hauptkardioide
    const double dx = x + 1.0;                    // Period-2 Bulb (r=0.25)
    if (dx*dx + y*y < 0.0625) return true;
    return false;
}

// NEU: NDC-Zentren-Cache pro Geometrie (tilesX, tilesY, width, height).
//      Vermeidet per-Frame tileIndexToPixelCenter-Aufrufe. (Bezug zu Schneefuchs)
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

// thread_local, um Re-Allokationen und false sharing zu vermeiden. (Bezug zu Schneefuchs)
thread_local NdcCenterCache g_ndcCache;

// thread_local Buffer – Allokationen vermeiden. (Bezug zu Schneefuchs)
thread_local std::vector<float> g_bufE;
thread_local std::vector<float> g_bufC;
thread_local std::vector<float> g_bufS;

// Stats-Cache & Cadence (alle kStatsEvery Frames neu) (Bezug zu Schneefuchs)
thread_local int   g_statsTick = 0;
thread_local int   g_statsN    = -1;
thread_local float g_eMed = 0.0f, g_eMad = 1.0f;
thread_local float g_cMed = 0.0f, g_cMad = 1.0f;

// kleiner, funktion-lokaler Zustand für Hysterese/Lock & Retarget (kein Header-Touch)
static int s_lockLeft = 0;
static int s_sinceRetarget = 0;
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

    // Warm-up: Richtung einfrieren (Otter)
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

    // Warm-up Early-Exit (Otter)
    if (freezeDirection) {
        out.shouldZoom = true;
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3][WARMUP] freeze-direction t=%.2fs (limit=%.2fs)",
                           warmupSec, kNO_TURN_WARMUP_SEC);
        }
        return out;
    }

    // Retarget nur alle N Frames (nur bei AlwaysZoom)
    if (Settings::ForceAlwaysZoom) {
        if (++s_sinceRetarget < kRetargetInterval) {
            out.shouldZoom = true;           // Zoom läuft, Richtung bleibt
            out.newOffset   = previousOffset;
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] skip_retarget interval=%d/%d",
                               s_sinceRetarget, kRetargetInterval);
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

    // Statistik: nur alle kStatsEvery Frames oder wenn N wechselt (Schneefuchs)
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

    // 1. Pass: Scores & Summen (Schneefuchs) – parallel optional
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

    // bestScore/bestIdx separat (serielle O(N) reicht; ggf. SIMD)
    float bestScore = -1e9f;
    int   bestIdx   = -1;
    for (int i = 0; i < N; ++i) {
        const float si = g_bufS[i];
        if (si > bestScore) { bestScore = si; bestIdx = i; }
    }

    // Signalstärke via Z-Score-Streuung
    (void)bestIdx; // may be overwritten by no-black pass
    const double meanS = sumS / std::max(1, N);
    const double varS  = std::max(0.0, (sumS2 / std::max(1, N)) - meanS * meanS);
    const double stdS  = std::sqrt(varS);
    const bool   hasSignal = (stdS >= kMIN_SIGNAL_Z);

    // Early-Exit bei "kein Signal" (wenn nicht AlwaysZoom) (Otter)
    if (!hasSignal && !Settings::ForceAlwaysZoom) {
        out.shouldZoom = false;
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3] no_signal early-exit stdS=%.3f (thr=%.3f)", (float)stdS, kMIN_SIGNAL_Z);
        }
        return out;
    }

    // Softmax-Temperatur adaptiv
    float temp = kTEMP_BASE;
    if (stdS > 1e-6) {
        temp = static_cast<float>(kTEMP_BASE / (0.5f + (float)stdS)); // 0.5 vermeidet extremes Einfrieren
    }
    temp = clampf(temp, 0.2f, 2.5f);

    // Softmax-Sparsification (Schneefuchs)
    const float sMax      = bestScore;
    const float sCutScore = sMax + temp * kSOFTMAX_LOG_EPS;

    // Precompute invZoom (Divisionen sparen)
    const double invZoom = 1.0 / (double)zoom;

    // 2. Pass: Normierter Schwerpunkt (nur signifikante Terme),
    //          HARTE No-Black-Filterung (Cardioid/2er-Bulb) (Otter/Schneefuchs)
    double sumW = 0.0, numX = 0.0, numY = 0.0;
    int    interiorSkipped = 0;

    const float invTempF = 1.0f / std::max(1e-6f, temp);

#ifdef ZOOMLOGIC_OMP
#pragma omp parallel for reduction(+:sumW,numX,numY,interiorSkipped) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        const float si = g_bufS[i];
        if (si < sCutScore) continue;

        // Komplexe Koordinaten des Kachelzentrums (NDC -> Complex via Offset/Zoom)
        const double cx = (double)currentOffset.x + (double)g_ndcCache.ndcX[i] * invZoom;
        const double cy = (double)currentOffset.y + (double)g_ndcCache.ndcY[i] * invZoom;

        if (isInsideCardioidOrBulb(cx, cy)) {
            ++interiorSkipped;
            continue; // harter Ausschluss von Innenbereichen
        }

        // Softmax-Gewicht (float expf für Speed, Summen in double für Stabilität)
        const float sShift = (si - sMax) * invTempF;
        const float exf    = std::expf(sShift);
        const double w     = (double)exf;
        sumW += w;
        numX += w * (double)g_ndcCache.ndcX[i];
        numY += w * (double)g_ndcCache.ndcY[i];
    }

    // Bestes "adj" (ohne Innenkandidaten) seriell bestimmen (klar & stabil)
    float bestAdjScore = -1e9f;
    int   bestAdjIdx   = -1;
    for (int i = 0; i < N; ++i) {
        const float si = g_bufS[i];
        if (si < sCutScore) continue;
        const double cx = (double)currentOffset.x + (double)g_ndcCache.ndcX[i] * invZoom;
        const double cy = (double)currentOffset.y + (double)g_ndcCache.ndcY[i] * invZoom;
        if (isInsideCardioidOrBulb(cx, cy)) continue;
        if (si > bestAdjScore) { bestAdjScore = si; bestAdjIdx = i; }
    }

    ZoomResult& o = out; // Alias
    if (bestAdjIdx >= 0) {
        o.bestIndex    = bestAdjIdx;
        o.bestScore    = bestAdjScore;
        o.bestEntropy  = entropy[bestAdjIdx];
        o.bestContrast = contrast[bestAdjIdx];
    } else {
        // Falls alles rausfiel: fallback auf initiale Auswahl (wird gleich durch Hysterese stabilisiert)
        o.bestIndex = bestIdx; o.bestScore = bestScore;
        o.bestEntropy  = (bestIdx >= 0) ? entropy[bestIdx]  : 0.0f;
        o.bestContrast = (bestIdx >= 0) ? contrast[bestIdx] : 0.0f;
    }

    double ndcX = 0.0, ndcY = 0.0;
    if (sumW > 0.0) {
        const double inv = 1.0 / sumW;
        ndcX = numX * inv;
        ndcY = numY * inv;
    }
    // Fallback: (0,0), falls alles unter Schwelle/innen.

    // Relative Hysterese + Lock auf Zielwechsel (vor Bewegung)
    if (state.lastAcceptedIndex >= 0 && o.bestIndex >= 0 && o.bestIndex != state.lastAcceptedIndex) {
        if (s_lockLeft > 0) {
            --s_lockLeft;
            o.bestIndex    = state.lastAcceptedIndex;
            o.bestScore    = state.lastAcceptedScore;
            o.bestEntropy  = entropy[o.bestIndex];
            o.bestContrast = contrast[o.bestIndex];
            if (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZOOMV3] lock_hold last=%d left=%d", o.bestIndex, s_lockLeft);
            }
        } else {
            const double rel = (state.lastAcceptedScore > 1e-12f)
                ? (double(o.bestScore) / double(state.lastAcceptedScore)) : 1e9;
            if (rel < (1.0 + kHYST_REL)) {
                // nicht deutlich besser: alten beibehalten
                o.bestIndex    = state.lastAcceptedIndex;
                o.bestScore    = state.lastAcceptedScore;
                o.bestEntropy  = entropy[o.bestIndex];
                o.bestContrast = contrast[o.bestIndex];
                if (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ZOOMV3] hysteresis_keep last=%d rel=%.3f thr=%.3f",
                                   state.lastAcceptedIndex, rel, 1.0 + kHYST_REL);
                }
            } else {
                s_lockLeft = kLOCK_FRAMES; // Wechsel akzeptiert -> kurz sperren
                if (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ZOOMV3] switch_accept old=%d new=%d rel=%.3f lock=%d",
                                   state.lastAcceptedIndex, o.bestIndex, rel, s_lockLeft);
                }
            }
        }
    }

    const float2 proposedOffset = make_float2(
        currentOffset.x + (float)(ndcX * invZoom),
        currentOffset.y + (float)(ndcY * invZoom)
    );

    const float dx = proposedOffset.x - previousOffset.x;
    const float dy = proposedOffset.y - previousOffset.y;
    const float dist = std::sqrt(dx*dx + dy*dy);

    // Bewegung glätten (EMA, adaptiv nur nach Distanz)
    float emaAlpha = kEMA_ALPHA_MIN + (kEMA_ALPHA_MAX - kEMA_ALPHA_MIN) * clampf(dist / 0.5f, 0.0f, 1.0f);
    if (Settings::ForceAlwaysZoom && !hasSignal) {
        // Otter: leichter Drift auch ohne klares Signal
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
        state.cooldownLeft      = 0; // V3: kein Cooldown-Konzept mehr
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
            "[ZOOMV3] move: dist=%.4f emaAlpha=%.3f ndc=(%.4f,%.4f) propOff=(%.5f,%.5f) newOff=(%.5f,%.5f) tiles=(%d,%d) ms=%.3f",
            dist, emaAlpha, (float)ndcX, (float)ndcY,
            proposedOffset.x, proposedOffset.y,
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
