// MAUS: Perf V3 hot path – no header/API change (ASCII logs only)

// 🐭 Maus: Zoom V3 – kontinuierlicher Schwerpunkt, glatter Drift, deterministisch.
// 🦦 Otter: Warm-up ohne Richtungswechsel: erst zoomen, dann lenken. (Bezug zu Otter)
// 🦊 Schneefuchs: Minimalinvasiv, keine Header-/API-Änderung. ASCII-Logs. (Bezug zu Schneefuchs)

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

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
// 🦦 Otter: sanfter Drift auch ohne starkes Signal, wenn AlwaysZoom aktiv ist
static constexpr float kFORCE_MIN_DRIFT_ALPHA = 0.05f;

// 🟢 NEU: Warm-up-Zeit (Sekunden), in der KEIN Richtungswechsel erfolgt.
//         Zoom läuft weiter, aber Offset bleibt unverändert. (Bezug zu Otter)
static constexpr double kNO_TURN_WARMUP_SEC = 2.0;

// 🟢 NEU: Softmax-Sparsification – ignoriere Beiträge mit sehr kleiner Gewichtung.
//         si < sMax + temp * kSOFTMAX_LOG_EPS -> Beitrag vernachlässigbar.
//         exp(-7) ≈ 0.0009, ausreichend klein für Schwerpunkt. (Bezug zu Schneefuchs)
static constexpr float kSOFTMAX_LOG_EPS = -7.0f;

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

// 🟢 NEU: MAD in-place auf wiederverwendetem Buffer; keine zusätzlichen Allokationen.
//         (Bezug zu Schneefuchs)
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

// 🟢 NEU: NDC-Zentren-Cache pro Geometrie (tilesX, tilesY, width, height).
//         Vermeidet per-Frame tileIndexToPixelCenter-Aufrufe. (Bezug zu Schneefuchs)
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

// thread_local Buffer für robuste Statistik – Allokationen vermeiden. (Bezug zu Schneefuchs)
thread_local std::vector<float> g_bufE;
thread_local std::vector<float> g_bufC;
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

// V3: Kontinuierlicher Schwerpunkt (Softmax) + EMA-Glättung
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

    // ── Warm-up-Timer: ab erstem Aufruf läuft die Uhr. ───────────────────────
    static bool warmupInit = false; // Schneefuchs: static, funktional lokal
    static clock::time_point warmupStart;
    if (!warmupInit) { warmupStart = t0; warmupInit = true; }
    const double warmupSec = std::chrono::duration<double>(t0 - warmupStart).count();
    const bool freezeDirection = (warmupSec < kNO_TURN_WARMUP_SEC);

    ZoomResult out;
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

    // 🟢 NEU: Warm-up vor schwerer Arbeit – sofortiger Early-Exit, Richtung einfrieren.
    //         Zoom bleibt extern aktiv. (Bezug zu Otter)
    if (freezeDirection) {
        out.shouldZoom = true;
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV3][WARMUP] freeze-direction t=%.2fs (limit=%.2fs)",
                           warmupSec, kNO_TURN_WARMUP_SEC);
        }
        return out;
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

    // 🟢 NEU: NDC-Zentren cachen (geometrieabhängig). (Bezug zu Schneefuchs)
    g_ndcCache.ensure(tilesX, tilesY, width, height);

    // 🟢 NEU: Robuste Statistik ohne frische Allokationen. (Bezug zu Schneefuchs)
    g_bufE.assign(entropy.begin(), entropy.begin() + N);
    g_bufC.assign(contrast.begin(), contrast.begin() + N);

    const float e_med = median_inplace(g_bufE);
    const float e_mad = mad_inplace_from_center(g_bufE, e_med);
    const float c_med = median_inplace(g_bufC);
    const float c_mad = mad_inplace_from_center(g_bufC, c_med);

    // Z-Scores + lineare Kombination – 1. Pass: Statistik, Bestes, Varianz
    float bestScore = -1e9f;
    int   bestIdx   = -1;
    double sumS  = 0.0;
    double sumS2 = 0.0;

    for (int i = 0; i < N; ++i) {
        const float ez = (entropy[i]  - e_med) / (e_mad <= 1e-6f ? 1.0f : e_mad);
        const float cz = (contrast[i] - c_med) / (c_mad <= 1e-6f ? 1.0f : c_mad);
        const float si = kALPHA_E * ez + kBETA_C * cz;
        sumS  += si;
        sumS2 += (double)si * (double)si;
        if (si > bestScore) { bestScore = si; bestIdx = i; }
    }

    // Signalstärke via Z-Score-Spanne
    const double meanS = sumS / std::max(1, N);
    const double varS  = std::max(0.0, (sumS2 / std::max(1, N)) - meanS * meanS);
    const double stdS  = std::sqrt(varS);
    const bool hasSignal = (stdS >= kMIN_SIGNAL_Z);

    out.bestScore    = bestScore;
    out.bestEntropy  = (bestIdx >= 0) ? entropy[bestIdx]  : 0.0f;
    out.bestContrast = (bestIdx >= 0) ? contrast[bestIdx] : 0.0f;
    out.bestIndex    = bestIdx;

    // Softmax-Temperatur adaptiv
    float temp = kTEMP_BASE;
    if (stdS > 1e-6) {
        temp = static_cast<float>(kTEMP_BASE / (0.5 + stdS)); // 0.5 vermeidet extremes Einfrieren
    }
    temp = clampf(temp, 0.2f, 2.5f);

    // 🟢 NEU: Softmax-Sparsification: nur relevante Beiträge exponentieren.
    //         Schwelle in Score-Domäne: s >= sMax + temp * kSOFTMAX_LOG_EPS. (Bezug zu Schneefuchs)
    const float sMax      = bestScore;
    const float sCutScore = sMax + temp * kSOFTMAX_LOG_EPS;

    // 2. Pass: Normierter Schwerpunkt in NDC – nur signifikante Terme
    double sumW = 0.0;
    double numX = 0.0;
    double numY = 0.0;

    for (int i = 0; i < N; ++i) {
        const float ez = (entropy[i]  - e_med) / (e_mad <= 1e-6f ? 1.0f : e_mad);
        const float cz = (contrast[i] - c_med) / (c_mad <= 1e-6f ? 1.0f : c_mad);
        const float si = kALPHA_E * ez + kBETA_C * cz;
        if (si < sCutScore) continue; // Otter: cheap skip

        const double ex = std::exp((si - sMax) / std::max(1e-6f, temp));
        sumW += ex;
        numX += ex * (double)g_ndcCache.ndcX[i];
        numY += ex * (double)g_ndcCache.ndcY[i];
    }

    double ndcX = 0.0;
    double ndcY = 0.0;
    if (sumW > 0.0) {
        const double inv = 1.0 / sumW;
        ndcX = numX * inv;
        ndcY = numY * inv;
    }
    // Otter: fallback bleibt (0,0), falls alle Beiträge unterhalb Schwelle liegen.

    const float2 proposedOffset = make_float2(
        currentOffset.x + static_cast<float>(ndcX) / zoom,
        currentOffset.y + static_cast<float>(ndcY) / zoom
    );

    const float dx = proposedOffset.x - previousOffset.x;
    const float dy = proposedOffset.y - previousOffset.y;
    const float dist = std::sqrt(dx*dx + dy*dy);

    // Bewegung glätten (EMA, adaptiv nur nach Distanz)
    float emaAlpha = kEMA_ALPHA_MIN + (kEMA_ALPHA_MAX - kEMA_ALPHA_MIN) * clampf(dist / 0.5f, 0.0f, 1.0f);
    if (Settings::ForceAlwaysZoom && !hasSignal) {
        // 🦦 Otter: leichter Drift auch ohne klares Signal
        emaAlpha = std::max(emaAlpha, kFORCE_MIN_DRIFT_ALPHA);
    }

    const float2 smoothed = make_float2(
        previousOffset.x * (1.0f - emaAlpha) + proposedOffset.x * emaAlpha,
        previousOffset.y * (1.0f - emaAlpha) + proposedOffset.y * emaAlpha
    );

    // ── Normale Ausgabe nach Warm-up ────────────────────────────────────────
    out.distance   = dist;
    out.newOffset  = (hasSignal || Settings::ForceAlwaysZoom) ? smoothed : previousOffset;
    out.shouldZoom = (hasSignal || Settings::ForceAlwaysZoom);

    // Kompatibilitäts-State
    const bool indexChanged = (bestIdx != state.lastAcceptedIndex);
    out.isNewTarget = indexChanged && hasSignal;
    if (hasSignal) {
        const bool first = (state.lastAcceptedIndex < 0);
        state.lastAcceptedIndex = bestIdx;
        state.lastAcceptedScore = first ? bestScore
                                        : (0.98f * state.lastAcceptedScore + 0.02f * bestScore);
        state.lastOffset        = out.newOffset;
        state.lastTilesX        = tilesX;
        state.lastTilesY        = tilesY;
        state.cooldownLeft      = 0; // V3: kein Cooldown-Konzept mehr
    }

    // Logging (ASCII only)
    if (Settings::debugLogging) {
        const int bx = (bestIdx >= 0 && tilesX > 0) ? bestIdx % tilesX : -1;
        const int by = (bestIdx >= 0 && tilesX > 0) ? bestIdx / tilesX : -1;
        const auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LUCHS_LOG_HOST(
            "[ZOOMV3] best=%d tile=(%d,%d) e=%.4f c=%.4f score=%.3f | meanS=%.3f stdS=%.3f temp=%.3f signal=%d",
            bestIdx, bx, by, out.bestEntropy, out.bestContrast, bestScore,
            (float)meanS, (float)stdS, temp, hasSignal ? 1 : 0
        );
        LUCHS_LOG_HOST(
            "[ZOOMV3] move: dist=%.4f emaAlpha=%.3f ndc=(%.4f,%.4f) propOff=(%.5f,%.5f) newOff=(%.5f,%.5f) tiles=(%d,%d) ms=%.3f",
            dist, emaAlpha, (float)ndcX, (float)ndcY,
            proposedOffset.x, proposedOffset.y,
            out.newOffset.x, out.newOffset.y,
            tilesX, tilesY, ms
        );
    }

    return out;
}

} // namespace ZoomLogic
