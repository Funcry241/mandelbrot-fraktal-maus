// Datei: src/zoom_logic.cpp
// Maus-Kommentar: Alpha 49 "Pinguin" - sanftes, kontinuierliches Zoomen ohne Elefant!
// + AutoTune: misst mehrere alpha‑Kandidaten, wählt den besten und loggt zyklisch.
// Ziel: Keine Rebuild-Orgie, sinnvolle Werte automatisch finden.
//
// EINSTELLBARE PARAMETER (in dieser Datei, ohne JSON):
//   kAUTO_TUNE_ENABLED      [bool]   true/false -> Auto‑Tuner an/aus
//   kCANDIDATE_ALPHAS       [Liste]  sinnvolle alpha‑Kandidaten (0.05–0.30 üblich)
//   kFRAMES_PER_CANDIDATE   [int]    Messdauer pro Kandidat (15–90 Frames; Standard 45)
//   kW_PROGRESS, kW_JERK,
//   kW_SWITCH, kW_BLACK     [float]  Gewichte des Rewards (0–2 sinnvoll)
//   kBLACK_ENTROPY_THRESH   [float]  Schwellwert für „schwarze Fläche“ (0.01–0.10)
//
// Hinweise zu Ranges:
//   • alpha:      0.05–0.30 (größer = schnelleres Nachführen, aber potenziell „hektisch“)
//   • Frames/Kandidat: 30–60 reichen oft; bei stark schwankenden FPS höher gehen
//   • Entropy-Black-Threshold: 0.02–0.06 passt oft; höher = vorsichtiger beim „Schwarz“
//
// Log-Ausgabe (wenn AutoTune aktiv):
//   [AutoTune] round=3 bestAlpha=0.160 avgFPS≈23.1 reward=+0.123 (kept from 5)
//   Die Zeile erscheint am Ende jeder Tuning-Runde (alle Kandidaten einmal gemessen).

#include "zoom_logic.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

// -----------------------------
//  Auto‑Tune Schalter & Parameter
// -----------------------------
static bool  kAUTO_TUNE_ENABLED    = true;   // <— EIN/AUS
// Kandidatenliste (du kannst hier frei „spielen“)
static float kCANDIDATE_ALPHAS[]   = { 0.08f, 0.12f, 0.16f, 0.20f, 0.24f, 0.28f };
// Messdauer pro Kandidat (Frames). 45 ≈ ~2 s bei 20–25 FPS.
static int   kFRAMES_PER_CANDIDATE = 45;

// Reward‑Gewichte (feintunen, falls nötig)
static float kW_PROGRESS           = 1.00f;  // höher => schneller reinzoomen wird stärker belohnt
static float kW_JERK               = 0.60f;  // höher => ruckartige Bewegungen werden stärker bestraft
static float kW_SWITCH             = 0.40f;  // höher => häufige Zielwechsel werden stärker bestraft
static float kW_BLACK              = 0.80f;  // höher => „schwarze Frames“ werden stärker bestraft

// „Schwarze Fläche“ (nahe 0 Entropie) -> Penalty
static float kBLACK_ENTROPY_THRESH = 0.04f;  // 0.02–0.06 ist sinnvoll

// Fallback, wenn Auto‑Tuner aus ist:
static float kFIXED_ALPHA          = 0.16f;  // 0.10–0.20: gut glatter Gleitflug

// -----------------------------
//  Auto‑Tuner interner Zustand
// -----------------------------
struct Candidate {
    float alpha;
    double reward = 0.0;
    int frames    = 0;
};

struct AutoTuner {
    std::vector<Candidate> pool;
    int currIdx          = 0;   // aktuell gemessener Kandidat
    int round            = 0;   // wie oft alle Kandidaten gemessen wurden
    int framesPerCand    = 45;
    // Laufzeit‑Stats
    int targetSwitches   = 0;
    float lastSpeed      = 0.0f;

    // FPS Grobschätzung für Log
    int   frameCount     = 0;
    double tStartMs      = 0.0;

    AutoTuner() {
        framesPerCand = kFRAMES_PER_CANDIDATE;
        for (float a : kCANDIDATE_ALPHAS) pool.push_back({a, 0.0, 0});
        tStartMs = nowMs();
    }

    static double nowMs() {
        using namespace std::chrono;
        return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
    }

    // Aufruf wenn bestIndex != previousIndex
    void onTargetSwitch() { targetSwitches++; }

    // pro Frame Messupdate
    void update(float zoom, float prevZoom,
                float2 offset, float2 prevOffset,
                float maxEntropy)
    {
        if (pool.empty()) return;
        Candidate& c = pool[currIdx];

        // Metriken
        float progress = std::logf(std::max(zoom, 1.0f)) - std::logf(std::max(prevZoom, 1.0f));
        float dx = offset.x - prevOffset.x;
        float dy = offset.y - prevOffset.y;
        float speed = std::sqrt(dx*dx + dy*dy);
        float jerk  = std::fabs(speed - lastSpeed);
        lastSpeed   = speed;

        float blackPenalty = (maxEntropy < kBLACK_ENTROPY_THRESH) ? 1.0f : 0.0f;
        float switchRate   = static_cast<float>(targetSwitches) * 0.02f; // grob: „Switches pro ~Sekunde“

        double R =  kW_PROGRESS * progress
                  - kW_JERK     * jerk
                  - kW_SWITCH   * switchRate
                  - kW_BLACK    * blackPenalty;

        c.reward += R;
        c.frames += 1;
        frameCount++;

        if (c.frames >= framesPerCand) {
            // nächster Kandidat
            currIdx++;
            targetSwitches = 0;
            lastSpeed = 0.0f;
            if (currIdx >= static_cast<int>(pool.size())) {
                // Runde abgeschlossen -> besten wählen und behalten
                std::sort(pool.begin(), pool.end(),
                          [](const Candidate& a, const Candidate& b){ return a.reward > b.reward; });
                const Candidate& best = pool.front();

                // grobe FPS-Schätzung
                double elapsed = nowMs() - tStartMs;
                double fps = (elapsed > 0.0) ? (frameCount * 1000.0 / elapsed) : 0.0;

                LUCHS_LOG_HOST("[AutoTune] round=%d bestAlpha=%.3f avgFPS≈%.1f reward=%+.3f (kept from %zu)",
                               ++round, best.alpha, fps, best.reward, pool.size());

                // Successive‑Halving: obere Hälfte behalten (solide, schnell konvergent)
                if (pool.size() > 2) {
                    pool.resize(pool.size() / 2);
                }

                // Reset für nächste Runde
                for (auto& p : pool) { p.reward = 0.0; p.frames = 0; }
                currIdx = 0;
                frameCount = 0;
                tStartMs   = nowMs();
            }
        }
    }

    float currentAlpha() const {
        if (pool.empty()) return kFIXED_ALPHA;
        // Während des Messens: jeweils alpha des aktuellen Kandidaten
        return pool[std::min(currIdx, (int)pool.size()-1)].alpha;
    }

    float bestAlpha() const {
        if (pool.empty()) return kFIXED_ALPHA;
        return std::max_element(pool.begin(), pool.end(),
               [](const Candidate& a, const Candidate& b){ return a.reward < b.reward; })->alpha;
    }
};

// Ein globaler Tuner‑State (nur in dieser Übersetzungseinheit sichtbar)
static AutoTuner gTuner;

// Historie für „prev“-Werte (lokal in dieser Datei, unabhängig von anderen Modulen)
// Reihenfolge bewusst gewählt (Ausrichtung vermeiden).
struct Hist {
    float2 prevOffset  = {0.0f, 0.0f};  // zuerst: 8 Byte (2 × float)
    float  prevZoom    = 1.0f;          // danach: 4 Byte
    int    prevIndex   = -1;            // dann: 4 Byte → sauber ausgerichtet
} gHist;

} // namespace (anonym)

// -------------------------------------------------------------------------------------
// Ab hier: bestehende Zoom‑Logik, minimalinvasiv mit Auto‑Tune verwoben
// -------------------------------------------------------------------------------------
namespace ZoomLogic {

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
    auto t0 = std::chrono::high_resolution_clock::now();

    ZoomResult result;
    result.bestIndex   = -1;
    result.shouldZoom  = false;
    result.isNewTarget = false;
    result.newOffset   = currentOffset;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const std::size_t totalTiles = static_cast<std::size_t>(tilesX * tilesY);

    // Debug: Min/Max sammeln
    float minE =  1e9f, maxE = -1e9f;
    float minC =  1e9f, maxC = -1e9f;
    if (Settings::debugLogging) {
        for (std::size_t i = 0; i < totalTiles; ++i) {
            if (i >= entropy.size() || i >= contrast.size()) continue;
            float e = entropy[i], c = contrast[i];
            minE = std::min(minE, e); maxE = std::max(maxE, e);
            minC = std::min(minC, c); maxC = std::max(maxC, c);
        }
        LUCHS_LOG_HOST("[ZoomEval] Entropy: min=%.4f max=%.4f | Contrast: min=%.4f max=%.4f", minE, maxE, minC, maxC);
        if (maxE < kBLACK_ENTROPY_THRESH) {
            LUCHS_LOG_HOST("[Diag] Black‑Verdacht: maxEntropy=%.4f < thresh=%.4f (Frame wirkt dunkel/schwarz)",
                           maxE, kBLACK_ENTROPY_THRESH);
        }
    }

    // Bestes Tile per Score = E * (1 + C)
    float bestScore = -1.0f;
    for (std::size_t i = 0; i < totalTiles; ++i) {
        if (i >= entropy.size() || i >= contrast.size()) {
            LUCHS_LOG_HOST("[ZoomEval] Index %zu out of bounds (entropy=%zu, contrast=%zu)", i, entropy.size(), contrast.size());
            continue;
        }
        float e = entropy[i];
        float c = contrast[i];
        float score = e * (1.0f + c);
        if (score > bestScore) {
            bestScore = score;
            result.bestIndex    = static_cast<int>(i);
            result.bestEntropy  = e;
            result.bestContrast = c;
        }
    }

    if (result.bestIndex < 0) {
        if (Settings::debugLogging)
            LUCHS_LOG_HOST("[ZoomEval] No target found - bestScore=%.4f", bestScore);
        return result;
    }

    // Tile‑Zentrum in NDC
    int bx = result.bestIndex % tilesX;
    int by = result.bestIndex / tilesX;
    float2 tileCenter;
    tileCenter.x = (bx + 0.5f) * tileSize;
    tileCenter.y = (by + 0.5f) * tileSize;
    tileCenter.x = (tileCenter.x / width  - 0.5f) * 2.0f;
    tileCenter.y = (tileCenter.y / height - 0.5f) * 2.0f;

    float2 proposedOffset = make_float2(
        currentOffset.x + tileCenter.x / zoom,
        currentOffset.y + tileCenter.y / zoom
    );

    // Distanz nur informativ (für Logs / Telemetrie)
    float dx = proposedOffset.x - previousOffset.x;
    float dy = proposedOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    float prevScore = previousEntropy * (1.0f + previousContrast);
    float scoreGain = (prevScore > 0.0f) ? ((bestScore - prevScore) / prevScore) : 1.0f;

    result.isNewTarget = true;
    result.shouldZoom  = true;

    // -----------------------------
    //  Auto‑Tuner Hooks
    // -----------------------------
    bool targetSwitched = (result.bestIndex != gHist.prevIndex);
    if (kAUTO_TUNE_ENABLED) {
        if (targetSwitched) gTuner.onTargetSwitch();
        float maxEntropy = (Settings::debugLogging ? maxE : result.bestEntropy);
        gTuner.update(/*zoom     */ zoom,
                      /*prevZoom */ gHist.prevZoom,
                      /*offset   */ proposedOffset,
                      /*prevOff  */ gHist.prevOffset,
                      /*maxE     */ maxEntropy);
    }

    // alpha bestimmen
    float alpha = kAUTO_TUNE_ENABLED ? gTuner.currentAlpha()
                                     : kFIXED_ALPHA;

    // Optionale leichte Verstärkung bei gutem Score‑Gain (nur Anzeige/Diag, Verhalten bleibt quasi gleich)
    float alphaBeforeBoost = alpha;
    if (scoreGain > 0.25f) alpha = std::min(alpha * 1.15f, 0.35f);

    // LERP (Kolibri)
    result.newOffset = make_float2(
        previousOffset.x * (1.0f - alpha) + proposedOffset.x * alpha,
        previousOffset.y * (1.0f - alpha) + proposedOffset.y * alpha
    );

    result.distance = dist;
    result.minDistance = 0.02f; // konservativer Default
    result.relEntropyGain  = (result.bestEntropy > 0.0f && previousEntropy > 0.0f)
                             ? (result.bestEntropy - previousEntropy) / previousEntropy
                             : 1.0f;
    result.relContrastGain = (result.bestContrast > 0.0f && previousContrast > 0.0f)
                             ? (result.bestContrast - previousContrast) / previousContrast
                             : 1.0f;

    // Historie fortschreiben
    gHist.prevZoom   = zoom;
    gHist.prevOffset = result.newOffset;
    gHist.prevIndex  = result.bestIndex;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (Settings::debugLogging) {
        // Score-Zerlegung + Geometrie + Zielwechsel
        LUCHS_LOG_HOST("[Diag] bestScore=%.4f prevScore=%.4f gain=%.3f | targetSwitched=%d switches=%d",
                       bestScore, prevScore, scoreGain, targetSwitched ? 1 : 0,
                       kAUTO_TUNE_ENABLED ? gTuner.targetSwitches : -1);
        LUCHS_LOG_HOST("[Diag] bestTile=(%d,%d) NDC=(%.4f,%.4f) proposedOffset=(%.5f,%.5f)",
                       bx, by, tileCenter.x, tileCenter.y, proposedOffset.x, proposedOffset.y);
        LUCHS_LOG_HOST("[Diag] alpha=%.3f (cand=%.3f%s) | dist=%.4f",
                       alpha, alphaBeforeBoost, (alpha != alphaBeforeBoost ? " +boost" : ""), dist);

        LUCHS_LOG_HOST("[ZoomEval] i=%d E=%.2f C=%.2f d=%.4f g=%.2f a=%.3f Z | %.3fms",
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
