// Datei: src/zoom_logic.cpp
// 🐭 Maus-Kommentar: Alpha 49 "Pinguin" - sanftes, kontinuierliches Zoomen ohne Elefant! Ziel wird immer interpoliert verfolgt, Score fließt in Glättung ein. Kein Warten, kein Hüpfen. Schneefuchs genießt den Flug, Otter testet Stabilität.
// 🐼 Panda: Bewertet Entropie x (1 + Kontrast) als Zielscore.
// 🐦 Kolibri: Weiche Bewegung via LERP (Zoom ist Gleitflug).
// 🐍 Flugente: float2 bleibt für Performance aktiv.
// 🔬 Blaupause: Laufzeitmessung mit std::chrono – erkennt Zoomlogik-Overhead.

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <cmath>
#include <chrono>

namespace ZoomLogic {

// 🐭 Maus: Lokale Konfigurationsparameter für Zoomverhalten – einstellbar zum Testen.
// 🦦 Otter: Jeder Parameter hat klaren Effektbereich, direkt in der Datei testbar.
// 🐑 Schneefuchs: Später über settings.hpp externalisierbar.

constexpr float ZOOM_STRENGTH     = 1.2f;
// Verstärkt die Rohverschiebung des Ziels. 
// 1.0 = normale Verschiebung, 1.5 = aggressiv, 2.0+ = stark springend
// Empfehlung: 1.2 – 1.8

constexpr float CONTRAST_WEIGHT  = 0.5f;
// Gewichtung des Kontrasts im Score.
// 1.0 = gleichwertig zu Entropie, 1.5 = Kontrast wichtiger, 2.0+ = Fokus auf harte Unterschiede
// Empfehlung: 1.0 – 2.0

constexpr float ALPHA_LERP_MIN   = 0.08f;
constexpr float ALPHA_LERP_MAX   = 0.30f;
// Steuert das minimale/maximale LERP-Gewicht für Offset-Übernahme.
// MIN: bei kleinen Sprüngen (sanft), MAX: bei großen Gewinnen/Distanzen (schnell).
// Empfehlung: MIN 0.05 – 0.10, MAX 0.25 – 0.40

constexpr float DIST_NORM        = 0.05f;
// Normierung der Distanz in "Screen-Units" für Alpha-Berechnung.
// 0.05 = mittlere Bewegung → Alpha ≈ MAX
// Empfehlung: 0.03 – 0.10

constexpr float GAIN_NORM        = 0.20f;
// Normierung des relativen Score-Gewinns.
// 0.20 = ~20% mehr Score → Alpha ≈ MAX
// Empfehlung: 0.15 – 0.40

ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    float2 currentOffset,
    float zoom,
    int width,
    int height,
    int tileSize,
    float2 previousOffset,
    [[maybe_unused]] int previousIndex,
    float previousEntropy,
    float previousContrast
) {
    auto t0 = std::chrono::high_resolution_clock::now(); // 🔬 Startzeit

    ZoomResult result;
    result.bestIndex = -1;
    result.shouldZoom = false;
    result.isNewTarget = false;
    result.newOffset = currentOffset;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const std::size_t totalTiles = static_cast<std::size_t>(tilesX * tilesY);

    if (Settings::debugLogging) {
        float minE = 9999.0f, maxE = -9999.0f;
        float minC = 9999.0f, maxC = -9999.0f;
        for (std::size_t i = 0; i < totalTiles; ++i) {
            if (i >= entropy.size() || i >= contrast.size()) continue;
            float e = entropy[i];
            float c = contrast[i];
            if (e < minE) minE = e;
            if (e > maxE) maxE = e;
            if (c < minC) minC = c;
            if (c > maxC) maxC = c;
        }
        LUCHS_LOG_HOST("[ZoomEval] Entropy: min=%.4f max=%.4f | Contrast: min=%.4f max=%.4f", minE, maxE, minC, maxC);
    }

    float bestScore = -1.0f;

    // 🐭 Zugriffssicherheit – validiert Entropie-/Kontrastgröße vor Zugriff
    for (std::size_t i = 0; i < totalTiles; ++i) {
        if (i >= entropy.size() || i >= contrast.size()) {
            LUCHS_LOG_HOST("[ZoomEval] Index %zu out of bounds (entropy=%zu, contrast=%zu)", i, entropy.size(), contrast.size());
            continue;
        }

        float entropyVal  = entropy[i];
        float contrastVal = contrast[i];

        float score = entropyVal * (1.0f + contrastVal * CONTRAST_WEIGHT);  // 🐼 Otter: Kontrast betont
        if (score > bestScore) {
            bestScore = score;
            result.bestIndex    = static_cast<int>(i);
            result.bestEntropy  = entropyVal;
            result.bestContrast = contrastVal;
        }
    }

    if (result.bestIndex < 0) {
        if (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZoomEval] No target found - bestScore=%.4f", bestScore);
        return result;
    }

    int bx = result.bestIndex % tilesX;
    int by = result.bestIndex / tilesX;

    float2 tileCenter;
    tileCenter.x = (bx + 0.5f) * tileSize;
    tileCenter.y = (by + 0.5f) * tileSize;
    tileCenter.x = (tileCenter.x / width  - 0.5f) * 2.0f;
    tileCenter.y = (tileCenter.y / height - 0.5f) * 2.0f;

    float2 proposedOffset = make_float2(
        currentOffset.x + (tileCenter.x / zoom) * ZOOM_STRENGTH,
        currentOffset.y + (tileCenter.y / zoom) * ZOOM_STRENGTH
    );

    float dx = proposedOffset.x - previousOffset.x;
    float dy = proposedOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    float prevScore = previousEntropy * (1.0f + previousContrast);
    float scoreGain = (prevScore > 0.0f) ? ((bestScore - prevScore) / prevScore) : 1.0f;

    result.isNewTarget  = true;
    result.shouldZoom   = true;

    // 🐦 Kolibri: Dynamisches Alpha – reagiert auf Distanz und Score-Gewinn
    const float distNorm = std::min(dist / DIST_NORM, 1.0f);
    const float gainNorm = std::min(std::max(scoreGain, 0.0f) / GAIN_NORM, 1.0f);
    const float drive    = std::clamp(0.5f * distNorm + 0.5f * gainNorm, 0.0f, 1.0f);
    const float alpha    = ALPHA_LERP_MIN + (ALPHA_LERP_MAX - ALPHA_LERP_MIN) * drive;

    result.newOffset = make_float2(
        previousOffset.x * (1.0f - alpha) + proposedOffset.x * alpha,
        previousOffset.y * (1.0f - alpha) + proposedOffset.y * alpha
    );

    result.distance = dist;
    result.minDistance = Settings::MIN_JUMP_DISTANCE;
    result.relEntropyGain  = (result.bestEntropy > 0.0f && previousEntropy > 0.0f)
                             ? (result.bestEntropy - previousEntropy) / previousEntropy
                             : 1.0f;
    result.relContrastGain = (result.bestContrast > 0.0f && previousContrast > 0.0f)
                             ? (result.bestContrast - previousContrast) / previousContrast
                             : 1.0f;

    auto t1 = std::chrono::high_resolution_clock::now(); // 🔬 Endzeit
    auto ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZoomEval] i=%d E=%.2f C=%.2f d=%.4f g=%.2f a=%.2f Z | %.3fms",
            result.bestIndex,
            result.bestEntropy,
            result.bestContrast,
            dist,
            scoreGain,
            alpha,
            ms
        );
    }

    return result;
}

} // namespace ZoomLogic
