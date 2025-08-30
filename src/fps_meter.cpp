#include "fps_meter.hpp"
#include <atomic>
#include <algorithm>
#include <cmath>

// 🦦 Otter: EMA smoothing keeps HUD stable without lag. (Bezug zu Otter)
// 🦊 Schneefuchs: atomics for lock-free snapshots; no iostream, ASCII-only. (Bezug zu Schneefuchs)

namespace {
    // Exponential moving average of core-ms
    static std::atomic<double> gEmaCoreMs{0.0};
    static std::atomic<bool>   gInit{false};

    constexpr double kAlpha = 0.20;     // smoothing factor
    constexpr double kBeta  = 1.0 - kAlpha; // 🦊 Schneefuchs: precompute complement to avoid repeated subtractions.
    constexpr double kEps   = 1e-6;     // avoid div-by-zero
    constexpr int    kClamp = 9999;     // sanity clamp for HUD
}

namespace FpsMeter {

void updateCoreMs(double coreMs) {
    // 🦊 Schneefuchs: Robustheit – ignoriere NaN/Inf; clamp gegen Negativwerte.
    if (!std::isfinite(coreMs)) return;
    coreMs = std::max(coreMs, 0.0);

    if (!gInit.load(std::memory_order_relaxed)) {
        gEmaCoreMs.store(coreMs > kEps ? coreMs : 0.0, std::memory_order_relaxed);
        gInit.store(true, std::memory_order_relaxed);
        return;
    }

    const double prev = gEmaCoreMs.load(std::memory_order_relaxed);
    const double ema  = (prev <= 0.0) ? coreMs : (kAlpha * coreMs + kBeta * prev);
    gEmaCoreMs.store(ema, std::memory_order_relaxed);
}

double currentMaxFps() {
    const double emaMs = gEmaCoreMs.load(std::memory_order_relaxed);
    if (!std::isfinite(emaMs) || emaMs <= kEps) return 0.0;
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
}

} // namespace FpsMeter
