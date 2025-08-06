// Datei: src/zoom_logic.cpp
// 🐭 Maus-Kommentar: Alpha 49 "Pinguin" – sanftes, kontinuierliches Zoomen ohne Elefant!
// 🦦 Otter: AutoTune + neue Metrik – bewertet jetzt nicht nur Entropie+Kontrast, sondern straft große homogene Flächen ab und boostet Detailkanten.
// 🐑 Schneefuchs: deterministisch, wirkt in allen Pfaden (Zoom, AutoTune, Overlay).
// 🦎 Chamäleon: erkennt TileSize-Änderungen und "sprunghafte" Indexwechsel mit wenig Zugewinn.
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
#include "heatmap_utils.hpp"

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
static float kCANDIDATE_ALPHAS[]   = { 0.08f, 0.12f, 0.16f, 0.20f, 0.24f, 0.28f };
static int   kFRAMES_PER_CANDIDATE = 45;

static float kW_PROGRESS           = 1.00f;
static float kW_JERK               = 0.60f;
static float kW_SWITCH             = 0.40f;
static float kW_BLACK              = 0.80f;

static float kBLACK_ENTROPY_THRESH = 0.04f;
static float kFIXED_ALPHA          = 0.16f;

// Chamäleon: Schwellwert für "sprunghaften" Indexwechsel mit wenig Zugewinn
constexpr float kINDEX_JUMP_THRESH = 0.05f;

struct Candidate {
    float alpha;
    double reward = 0.0;
    int frames    = 0;
};

struct AutoTuner {
    std::vector<Candidate> pool;
    int currIdx          = 0;
    int round            = 0;
    int framesPerCand    = 45;
    int targetSwitches   = 0;
    float lastSpeed      = 0.0f;
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

    void onTargetSwitch() { targetSwitches++; }

    void update(float zoom, float prevZoom,
                float2 offset, float2 prevOffset,
                float maxEntropy)
    {
        if (pool.empty()) return;
        Candidate& c = pool[currIdx];

        float progress = std::logf(std::max(zoom, 1.0f)) - std::logf(std::max(prevZoom, 1.0f));
        float dx = offset.x - prevOffset.x;
        float dy = offset.y - prevOffset.y;
        float speed = std::sqrt(dx*dx + dy*dy);
        float jerk  = std::fabs(speed - lastSpeed);
        lastSpeed   = speed;

        float blackPenalty = (maxEntropy < kBLACK_ENTROPY_THRESH) ? 1.0f : 0.0f;
        float switchRate   = static_cast<float>(targetSwitches) * 0.02f;

        double R =  kW_PROGRESS * progress
                  - kW_JERK     * jerk
                  - kW_SWITCH   * switchRate
                  - kW_BLACK    * blackPenalty;

        c.reward += R;
        c.frames += 1;
        frameCount++;

        if (c.frames >= framesPerCand) {
            currIdx++;
            targetSwitches = 0;
            lastSpeed = 0.0f;
            if (currIdx >= static_cast<int>(pool.size())) {
                std::sort(pool.begin(), pool.end(),
                          [](const Candidate& a, const Candidate& b){ return a.reward > b.reward; });
                const Candidate& best = pool.front();

                double elapsed = nowMs() - tStartMs;
                double fps = (elapsed > 0.0) ? (frameCount * 1000.0 / elapsed) : 0.0;

                LUCHS_LOG_HOST("[AutoTune] round=%d bestAlpha=%.3f avgFPS≈%.1f reward=%+.3f (kept from %zu)",
                               ++round, best.alpha, fps, best.reward, pool.size());

                if (pool.size() > 2) {
                    pool.resize(pool.size() / 2);
                }

                for (auto& p : pool) { p.reward = 0.0; p.frames = 0; }
                currIdx = 0;
                frameCount = 0;
                tStartMs   = nowMs();
            }
        }
    }

    float currentAlpha() const {
        if (pool.empty()) return kFIXED_ALPHA;
        return pool[std::min(currIdx, (int)pool.size()-1)].alpha;
    }

    float bestAlpha() const {
        if (pool.empty()) return kFIXED_ALPHA;
        return std::max_element(pool.begin(), pool.end(),
               [](const Candidate& a, const Candidate& b){ return a.reward < b.reward; })->alpha;
    }
};

static AutoTuner gTuner;

struct Hist {
    float2 prevOffset  = {0.0f, 0.0f};
    float  prevZoom    = 1.0f;
    int    prevIndex   = -1;
    int    prevTileSize = -1;
} gHist;

} // namespace

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
    }

    // --- Neue Metrik: vermeidet langweilige Flächen ---
    float bestScore = -1.0f;
    for (std::size_t i = 0; i < totalTiles; ++i) {
        if (i >= entropy.size() || i >= contrast.size()) continue;

        float e = entropy[i];
        float c = contrast[i];

        float boredomPenalty = (e < 0.15f && c < 0.15f) ? 0.5f : 1.0f;
        float edgeBoost = (c > 0.6f) ? 1.2f : 1.0f;

        float score = e * (1.0f + c) * boredomPenalty * edgeBoost;

        if (score > bestScore) {
            bestScore = score;
            result.bestIndex    = static_cast<int>(i);
            result.bestEntropy  = e;
            result.bestContrast = c;
        }
    }
    // --------------------------------------------------

    if (result.bestIndex < 0) {
        return result;
    }

    auto [px, py] = tileIndexToPixelCenter(result.bestIndex, tilesX, tilesY, width, height);

    float2 tileCenter;
    tileCenter.x = static_cast<float>((px / width)  - 0.5) * 2.0f;
    tileCenter.y = static_cast<float>((py / height) - 0.5) * 2.0f;

    float2 proposedOffset = make_float2(
        currentOffset.x + tileCenter.x / zoom,
        currentOffset.y + tileCenter.y / zoom
    );

    float dx = proposedOffset.x - previousOffset.x;
    float dy = proposedOffset.y - previousOffset.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    float prevScore = previousEntropy * (1.0f + previousContrast);
    float scoreGain = (prevScore > 0.0f) ? ((bestScore - prevScore) / prevScore) : 1.0f;

    result.isNewTarget = true;
    result.shouldZoom  = true;

    bool targetSwitched = (result.bestIndex != gHist.prevIndex);
    if (kAUTO_TUNE_ENABLED) {
        if (targetSwitched) gTuner.onTargetSwitch();
        float maxEntropy = (Settings::debugLogging ? maxE : result.bestEntropy);
        gTuner.update(zoom, gHist.prevZoom, proposedOffset, gHist.prevOffset, maxEntropy);
    }

    float alpha = kAUTO_TUNE_ENABLED ? gTuner.currentAlpha() : kFIXED_ALPHA;
    float alphaBeforeBoost = alpha;
    if (scoreGain > 0.25f) alpha = std::min(alpha * 1.15f, 0.35f);

    result.newOffset = make_float2(
        previousOffset.x * (1.0f - alpha) + proposedOffset.x * alpha,
        previousOffset.y * (1.0f - alpha) + proposedOffset.y * alpha
    );

    result.distance = dist;
    result.minDistance = 0.02f;
    result.relEntropyGain  = (result.bestEntropy > 0.0f && previousEntropy > 0.0f)
                             ? (result.bestEntropy - previousEntropy) / previousEntropy
                             : 1.0f;
    result.relContrastGain = (result.bestContrast > 0.0f && previousContrast > 0.0f)
                             ? (result.bestContrast - previousContrast) / previousContrast
                             : 1.0f;

    // Historie aktualisieren
    bool tileSizeChanged = (tileSize != gHist.prevTileSize);
    gHist.prevTileSize = tileSize;
    gHist.prevZoom   = zoom;
    gHist.prevOffset = result.newOffset;
    gHist.prevIndex  = result.bestIndex;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (Settings::debugLogging) {
        int bx = result.bestIndex % tilesX;
        int by = result.bestIndex / tilesX;
        LUCHS_LOG_HOST("[ZoomEval] bestScore=%.4f prevScore=%.4f gain=%.3f | tile=(%d,%d) NDC=(%.4f,%.4f) offset=(%.5f,%.5f) alpha=%.3f (cand=%.3f%s) dist=%.4f ms=%.3f",
                       bestScore, prevScore, scoreGain,
                       bx, by, tileCenter.x, tileCenter.y,
                       proposedOffset.x, proposedOffset.y,
                       alpha, alphaBeforeBoost, (alpha != alphaBeforeBoost ? " +boost" : ""), dist, ms);

        // Chamäleon-Logzeile
        bool indexJump = targetSwitched && (std::fabs(scoreGain) < kINDEX_JUMP_THRESH);
        LUCHS_LOG_HOST("[Chamäleon] tileSizeChange=%d indexJump=%d scoreGain=%.3f",
                       tileSizeChanged ? 1 : 0,
                       indexJump ? 1 : 0,
                       scoreGain);
    }

    return result;
}

} // namespace ZoomLogic
