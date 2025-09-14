///// Otter: FPS-Meter – glättet Framezeiten mit Ramp-EMA und Spike-Dämpfung für ruhige HUD-Anzeige.
///// Schneefuchs: Lock-free Atomics; keine iostreams; ASCII-only; deterministisch.
///// Maus: Keine API-Änderungen; schnelle Konvergenz, dann stabil; NaN/Inf-Filter; minimale Includes.
///// Datei: src/fps_meter.cpp

#include "fps_meter.hpp"
#include <atomic>
#include <algorithm>
#include <cmath>

// 🦦 Otter: EMA smoothing keeps HUD stable without lag.
// 🦊 Schneefuchs: atomics for lock-free snapshots; no iostream, ASCII-only.

namespace {
    // Exponential moving average of "core" frame time in milliseconds.
    static std::atomic<double>        gEmaCoreMs{0.0};
    static std::atomic<bool>          gInit{false};
    static std::atomic<unsigned int>  gFrames{0};

    // Smoothing parameters: fast ramp for first frames, then stable.
    constexpr double       kAlphaFast    = 0.45;  // startup smoothing
    constexpr double       kAlphaStable  = 0.20;  // steady-state smoothing
    constexpr unsigned int kRampFrames   = 20;    // number of fast frames

    // Spike damping: bound new samples relative to previous EMA.
    constexpr double kSpikeMinRatio = 0.50;       // do not drop more than 50% in one step
    constexpr double kSpikeMaxRatio = 2.00;       // do not grow by more than 2x in one step

    constexpr double kEps   = 1e-6;               // avoid div-by-zero
    constexpr int    kClamp = 9999;               // sanity clamp for HUD
}

namespace FpsMeter {

void updateCoreMs(double coreMs) {
    // 🦊 Schneefuchs: Robustheit – ignore NaN/Inf; clamp negatives to zero.
    if (!std::isfinite(coreMs)) return;
    coreMs = std::max(coreMs, 0.0);

    // First sample: initialize EMA directly.
    if (!gInit.load(std::memory_order_relaxed)) {
        gEmaCoreMs.store(coreMs > kEps ? coreMs : 0.0, std::memory_order_relaxed);
        gFrames.store(1u, std::memory_order_relaxed);
        gInit.store(true, std::memory_order_relaxed);
        return;
    }

    const double prev = gEmaCoreMs.load(std::memory_order_relaxed);

    // Determine current alpha based on how many frames we've seen.
    const unsigned int n = gFrames.load(std::memory_order_relaxed);
    const double alpha   = (n < kRampFrames) ? kAlphaFast : kAlphaStable;
    const double beta    = 1.0 - alpha;

    // Spike damping: limit new sample relative to previous EMA.
    double sample = coreMs;
    if (prev > 0.0) {
        const double minAllowed = prev * kSpikeMinRatio;
        const double maxAllowed = prev * kSpikeMaxRatio;
        sample = std::clamp(sample, minAllowed, maxAllowed);
    }

    // Standard EMA update.
    const double ema = (prev <= 0.0) ? sample : (alpha * sample + beta * prev);
    gEmaCoreMs.store(ema, std::memory_order_relaxed);
    gFrames.store(n + 1u, std::memory_order_relaxed);
}

double currentMaxFps() {
    const double emaMs = gEmaCoreMs.load(std::memory_order_relaxed);
    if (!std::isfinite(emaMs) || emaMs <= kEps) return 0.0;

    // "Max FPS from core time" = 1000 ms / emaCoreMs.
    const double fps = 1000.0 / emaMs;
    return std::min(fps, static_cast<double>(kClamp));
}

int currentMaxFpsInt() {
    const double fps = currentMaxFps();
    return (fps <= 0.0) ? 0 : static_cast<int>(std::lround(fps));
}

void reset() {
    gEmaCoreMs.store(0.0, std::memory_order_relaxed);
    gInit.store(false,   std::memory_order_relaxed);
    gFrames.store(0u,    std::memory_order_relaxed);
}

} // namespace FpsMeter
