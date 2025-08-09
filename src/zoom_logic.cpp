// Datei: src/zoom_logic.cpp
// 🐭 Maus: Zoom V2 – simpel, deterministisch, ruhig. Alte Zöpfe abgeschnitten.
// 🦦 Otter: Hysterese + Cooldown + EMA‑Glättung, alle Entscheidungen geloggt. (Bezug zu Otter)
// 🦊 Schneefuchs: Tiles kommen explizit vom Aufrufer; kein implizites Rechnen hier. (Bezug zu Schneefuchs)

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

// ------- V2-Parameter (Defaults). Hinweis: bei Bedarf in settings.hpp verschieben. -------
// Scoring-Gewichte
static constexpr float kALPHA_E = 1.00f;   // Gewicht Entropie
static constexpr float kBETA_C  = 0.50f;   // Gewicht Kontrast
// Annahmeschwellen
static constexpr float kHYSTERESIS       = 0.08f; // 8% besser als zuletzt akzeptiertes Ziel
static constexpr float kACCEPT_THRESHOLD = 0.40f; // Mindestscore zum Akzeptieren (nach Norm.)
static constexpr int   kCOOLDOWN_FRAMES  = 14;    // Frames Sperre nach Zielwechsel
// Bewegung / Glättung
static constexpr float kEMA_ALPHA_BASE   = 0.16f; // Grund-Glättung Richtung Ziel
static constexpr float kEMA_ALPHA_MAX    = 0.30f; // Kappung je Frame
// Signaldetektion
static constexpr float kMIN_SIGNAL_SCORE = 0.15f; // darunter: Pause (kein Zoom)
// Deadzone
static constexpr float kMIN_DISTANCE     = 0.02f; // ~NDC/Zoom-Skala (wie bisher)

// --- robuste Statistik (Median/MAD) ---
static inline float median_inplace(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n/2, v.end());
    float m = v[n/2];
    if (!(n & 1)) {
        auto it = std::max_element(v.begin(), v.begin() + n/2);
        m = (*it + m) * 0.5f;
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

    // Geometrie prüfen
    const int totalTiles = tilesX * tilesY;
    if (tilesX <= 0 || tilesY <= 0 || totalTiles <= 0) {
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV2] invalid tiles: tilesX=%d tilesY=%d", tilesX, tilesY);
        }
        return out;
    }
    const int N = std::min<int>(totalTiles, std::min<int>((int)entropy.size(), (int)contrast.size()));
    if (N <= 0) {
        if (Settings::debugLogging) LUCHS_LOG_HOST("[ZOOMV2] empty metrics -> no zoom");
        return out;
    }

    // Kopien für robuste Statistik
    std::vector<float> e(entropy.begin(),  entropy.begin()  + N);
    std::vector<float> c(contrast.begin(), contrast.begin() + N);

    // Median & MAD
    float e_med = median_inplace(e);
    float c_med = median_inplace(c);
    float e_mad = mad_from(entropy, e_med);
    float c_mad = mad_from(contrast, c_med);
    if (e_mad <= 1e-6f) e_mad = 1.0f;
    if (c_mad <= 1e-6f) c_mad = 1.0f;

    // Scoring (α*E' + β*C')
    float bestScore = -1e9f;
    int   bestIdx   = -1;
    for (int i = 0; i < N; ++i) {
        float ez = (entropy[i]  - e_med) / e_mad;
        float cz = (contrast[i] - c_med) / c_mad;
        float s  = kALPHA_E * ez + kBETA_C * cz;
        if (s > bestScore) { bestScore = s; bestIdx = i; }
    }

    // Signalerkennung: zu schwach? -> Pause
    if (bestScore < kMIN_SIGNAL_SCORE) {
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMV2] paused: low signal (bestScore=%.3f < %.3f)", bestScore, kMIN_SIGNAL_SCORE);
        }
        out.bestScore = bestScore;
        return out;
    }

    // Hysterese/Cooldown gegen Springen
    const bool indexChanged = (bestIdx != state.lastAcceptedIndex);
    bool accept = false;
    if (state.cooldownLeft > 0) {
        // im Cooldown nur akzeptieren, wenn signifikant besser
        accept = (bestScore >= state.lastAcceptedScore * (1.0f + kHYSTERESIS * 2.0f));
        state.cooldownLeft -= 1;
    } else {
        // normaler Modus: Hysterese oder absolute Schwelle
        accept = (bestScore >= state.lastAcceptedScore * (1.0f + kHYSTERESIS)) || (bestScore >= kACCEPT_THRESHOLD);
    }

    // Tilezentrum -> NDC-Vektor -> Offset-Vorschlag
    auto center = tileIndexToPixelCenter(bestIdx, tilesX, tilesY, width, height);
    float2 ndc;
    ndc.x = static_cast<float>((center.first  / width)  - 0.5) * 2.0f;
    ndc.y = static_cast<float>((center.second / height) - 0.5) * 2.0f;

    float2 proposedOffset = make_float2(
        currentOffset.x + ndc.x / zoom,
        currentOffset.y + ndc.y / zoom
    );

    const float dx = proposedOffset.x - previousOffset.x;
    const float dy = proposedOffset.y - previousOffset.y;
    const float dist = std::sqrt(dx*dx + dy*dy);

    // Bewegung glätten (EMA)
    float emaAlpha = kEMA_ALPHA_BASE;
    if (dist > 0.2f) emaAlpha = std::min(kEMA_ALPHA_MAX, emaAlpha * 1.5f);
    if (dist < 0.02f) emaAlpha = std::min(emaAlpha, 0.10f);

    float2 smoothed = make_float2(
        previousOffset.x * (1.0f - emaAlpha) + proposedOffset.x * emaAlpha,
        previousOffset.y * (1.0f - emaAlpha) + proposedOffset.y * emaAlpha
    );

    // Ausgabe füllen
    out.bestIndex      = bestIdx;
    out.bestEntropy    = entropy[bestIdx];
    out.bestContrast   = contrast[bestIdx];
    out.bestScore      = bestScore;
    out.distance       = dist;
    out.relEntropyGain = 0.0f; // Legacy-Metriken aktuell nicht genutzt
    out.relContrastGain= 0.0f;

    if (accept) {
        out.isNewTarget = indexChanged;
        out.shouldZoom  = true;
        out.newOffset   = smoothed;

        // Zustand aktualisieren
        if (indexChanged) state.cooldownLeft = kCOOLDOWN_FRAMES;
        state.lastAcceptedIndex = bestIdx;
        state.lastAcceptedScore = std::max(bestScore, state.lastAcceptedScore);
        state.lastOffset        = out.newOffset;
        state.lastTilesX        = tilesX;
        state.lastTilesY        = tilesY;
    } else {
        // Kein Wechsel angenommen – konservativ: Position halten
        out.isNewTarget = false;
        out.shouldZoom  = false;
        out.newOffset   = previousOffset;
    }

    // Logging
    if (Settings::debugLogging) {
        int bx = bestIdx % tilesX, by = bestIdx / tilesX;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LUCHS_LOG_HOST(
            "[ZOOMV2] best=%d tile=(%d,%d) e=%.4f c=%.4f score=%.3f | lastScore=%.3f idxChanged=%d accept=%d cooldown=%d",
            bestIdx, bx, by, out.bestEntropy, out.bestContrast, bestScore,
            state.lastAcceptedScore, indexChanged ? 1 : 0, accept ? 1 : 0, state.cooldownLeft
        );
        LUCHS_LOG_HOST(
            "[ZOOMV2] move: dist=%.4f emaAlpha=%.3f ndc=(%.4f,%.4f) propOff=(%.5f,%.5f) newOff=(%.5f,%.5f) tiles=(%d,%d) ms=%.3f",
            dist, emaAlpha, ndc.x, ndc.y,
            proposedOffset.x, proposedOffset.y,
            out.newOffset.x, out.newOffset.y,
            tilesX, tilesY, ms
        );
    }

    return out;
}

} // namespace ZoomLogic
