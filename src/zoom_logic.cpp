// 🐭 Maus: Zoom V3 – kontinuierlicher Schwerpunkt, glatter Drift, deterministisch.
// 🦦 Otter: Softmax-Schwerpunkt statt Zielspringen; EMA-Glättung, ForceAlwaysZoom als sanfter Drift. (Bezug zu Otter)
// 🦊 Schneefuchs: Tiles/Geometrie kommen vom Aufrufer; keine impliziten Annahmen. (Bezug zu Schneefuchs)

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

// ------- V3‑Parameter (lokal, bewusst nicht in settings.hpp, um Hotfixe zu erleichtern) -------
// Scoring-Gewichte
static constexpr float kALPHA_E = 1.00f; // Gewicht Entropie
static constexpr float kBETA_C  = 0.50f; // Gewicht Kontrast
// Softmax
static constexpr float kTEMP_BASE = 1.00f; // Grundtemperatur für Softmax (wird adaptiv skaliert)
// Bewegung / Glättung
static constexpr float kEMA_ALPHA_MIN = 0.06f; // minimale Glättung
static constexpr float kEMA_ALPHA_MAX = 0.30f; // maximale Glättung
// Signaldetektion
static constexpr float kMIN_SIGNAL_Z  = 0.15f; // minimale Z‑Score‑Stärke für "aktives" Signal
// Deadzone (nur Dokumentation – Ausgabe in out.minDistance)
static constexpr float kMIN_DISTANCE  = 0.02f; // ~NDC/Zoom-Skala
// 🦦 Otter: sanfter Drift auch ohne starkes Signal, wenn AlwaysZoom aktiv ist
static constexpr float kFORCE_MIN_DRIFT_ALPHA = 0.05f;

// --- robuste Statistik (Median/MAD) ---
static inline float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n   = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if ((n & 1) == 0) {
        // Schneefuchs: korrektes Even‑N‑Handling via zweitem nth_element
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
        const float m2 = v[mid - 1];
        m = 0.5f * (m + m2);
    }
    return m;
}

static inline float mad_from(const std::vector<float>& v, float med) {
    if (v.empty()) return 1.0f;
    std::vector<float> dev; dev.reserve(v.size());
    for (float x : v) dev.push_back(std::fabs(x - med));
    std::vector<float> tmp = dev; // copy für nth_element
    float m = median_inplace(tmp);
    return (m > 1e-6f) ? m : 1.0f;
}

// --- kleine Helfer ---
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

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

    // Mittelwert der lokalen Kontraste (4‑Nachbarn)
    double acc = 0.0;
    int cnt = 0;
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int i = ty * tilesX + tx;
            float center = entropy[i];
            float sum = 0.0f; int n = 0;
            const int nx[4] = { tx-1, tx+1, tx,   tx };
            const int ny[4] = { ty,   ty,   ty-1, ty+1 };
            for (int k = 0; k < 4; ++k) {
                if (nx[k] < 0 || ny[k] < 0 || nx[k] >= tilesX || ny[k] >= tilesY) continue;
                int j = ny[k] * tilesX + nx[k];
                sum += std::fabs(entropy[j] - center);
                ++n;
            }
            if (n > 0) { acc += (sum / n); ++cnt; }
        }
    }
    return (cnt > 0) ? static_cast<float>(acc / cnt) : 0.0f;
}

// V3: Kontinuierlicher Schwerpunkt (Softmax) + EMA‑Glättung
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    int tilesX, int tilesY,
    int width, int height,
    float2 currentOffset, float zoom,
    float2 previousOffset,
    ZoomState& state) noexcept
{
    auto t0 = std::chrono::high_resolution_clock::now();

    ZoomResult out;
    out.bestIndex   = -1;
    out.shouldZoom  = false;
    out.isNewTarget = false;
    out.newOffset   = previousOffset; // default: keine Änderung
    out.minDistance = kMIN_DISTANCE;
    out.bestScore   = 0.0f;
    out.bestEntropy = 0.0f;
    out.bestContrast= 0.0f;

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

    // Kopien für robuste Statistik (auf N beschnitten)
    std::vector<float> e(entropy.begin(),  entropy.begin()  + N);
    std::vector<float> c(contrast.begin(), contrast.begin() + N);

    // Median & MAD (robuste Normierung)
    float e_med = median_inplace(e);
    float c_med = median_inplace(c);
    float e_mad = mad_from(e, e_med);
    float c_mad = mad_from(c, c_med);
    if (e_mad <= 1e-6f) e_mad = 1.0f;
    if (c_mad <= 1e-6f) c_mad = 1.0f;

    // Z‑Scores + lineare Kombination
    float bestScore = -1e9f;
    int   bestIdx   = -1;

    double sumS  = 0.0;
    double sumS2 = 0.0;
    std::vector<float> s; s.resize(N);
    for (int i = 0; i < N; ++i) {
        float ez = (entropy[i]  - e_med) / e_mad;
        float cz = (contrast[i] - c_med) / c_mad;
        float si = kALPHA_E * ez + kBETA_C * cz;
        s[i] = si;
        sumS  += si;
        sumS2 += (double)si * (double)si;
        if (si > bestScore) { bestScore = si; bestIdx = i; }
    }

    // Signalstärke via Z‑Score‑Spanne
    double meanS = sumS / std::max(1, N);
    double varS  = std::max(0.0, (sumS2 / std::max(1, N)) - meanS * meanS);
    double stdS  = std::sqrt(varS);
    const bool hasSignal = (stdS >= kMIN_SIGNAL_Z);

    out.bestScore    = bestScore;
    out.bestEntropy  = (bestIdx >= 0) ? entropy[bestIdx]  : 0.0f;
    out.bestContrast = (bestIdx >= 0) ? contrast[bestIdx] : 0.0f;
    out.bestIndex    = bestIdx;

    // Softmax‑Temperatur adaptiv
    float temp = kTEMP_BASE;
    if (stdS > 1e-6) {
        temp = static_cast<float>(kTEMP_BASE / (0.5 + stdS)); // 0.5 vermeidet extremes Einfrieren
    }
    temp = clampf(temp, 0.2f, 2.5f);

    // Softmax‑Gewichte (stabilisiert: shift um max)
    float sMax = bestScore;
    std::vector<float> w; w.resize(N);
    double sumW = 0.0;
    for (int i = 0; i < N; ++i) {
        double ex = std::exp((s[i] - sMax) / std::max(1e-6f, temp));
        w[i] = static_cast<float>(ex);
        sumW += ex;
    }
    const double invSumW = (sumW > 0.0) ? (1.0 / sumW) : 0.0;

    // Kontinuierliches Zielzentrum = gewichteter Schwerpunkt der Tile‑Zentren
    double ndcX = 0.0;
    double ndcY = 0.0;
    for (int i = 0; i < N; ++i) {
        const double wi = w[i] * invSumW;
        auto center = tileIndexToPixelCenter(i, tilesX, tilesY, width, height);
        const double cx = static_cast<double>(center.first)  / static_cast<double>(width);
        const double cy = static_cast<double>(center.second) / static_cast<double>(height);
        const double xN = (cx - 0.5) * 2.0; // NDC
        const double yN = (cy - 0.5) * 2.0;
        ndcX += wi * xN;
        ndcY += wi * yN;
    }

    float2 proposedOffset = make_float2(
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

    float2 smoothed = make_float2(
        previousOffset.x * (1.0f - emaAlpha) + proposedOffset.x * emaAlpha,
        previousOffset.y * (1.0f - emaAlpha) + proposedOffset.y * emaAlpha
    );

    // Ausgabe
    out.distance   = dist;
    out.newOffset  = (hasSignal || Settings::ForceAlwaysZoom) ? smoothed : previousOffset;
    out.shouldZoom = (hasSignal || Settings::ForceAlwaysZoom);

    // Kompatibilitäts‑State
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
        state.cooldownLeft      = 0; // V3: kein Cooldown‑Konzept mehr
    }

    // Logging (ASCII only)
    if (Settings::debugLogging) {
        int bx = (bestIdx >= 0 && tilesX > 0) ? bestIdx % tilesX : -1;
        int by = (bestIdx >= 0 && tilesX > 0) ? bestIdx / tilesX : -1;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

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
        if (Settings::ForceAlwaysZoom) {
            LUCHS_LOG_HOST(
                "[ZOOMV3] forceAlwaysZoom=1 -> shouldZoom=%d (signal=%d)",
                out.shouldZoom ? 1 : 0, hasSignal ? 1 : 0
            );
        }
    }

    return out;
}

} // namespace ZoomLogic
