///// Otter: FrameBudgetController – frame-time cap (fixed/adaptive EMA); works best with VSync OFF.
///// Schneefuchs: Log cadence 1/60 to keep noise low; deterministic, measurable, ASCII-only logs.
///// Maus: No hidden timing changes; sleep uses coarse then yield-based fine wait.
///// Datei: src/frame_budget_controller.hpp

#pragma once
#include <atomic>
#include <chrono>
#include <thread>
#include "luchs_log_host.hpp"

namespace otterdream {

class FrameBudgetController {
public:
    FrameBudgetController(double targetMs = 16.666, bool adaptive = false) noexcept
        : targetMs_(targetMs), adaptive_(adaptive) {}

    void setTargetMs(double ms) noexcept { targetMs_ = (ms > 1.0 ? ms : 1.0); }
    void setAdaptive(bool on) noexcept { adaptive_ = on; }

    void onFrameEnd(double measuredMs) noexcept {
        ++frameCounter_;
        lastMeasuredMs_ = (measuredMs > 0.0 ? measuredMs : 0.0);
        const double alpha = 0.15;
        if (emaMs_ <= 0.0) emaMs_ = lastMeasuredMs_;
        else               emaMs_ = (1.0 - alpha) * emaMs_ + alpha * lastMeasuredMs_;
        if (adaptive_) {
            const double next = emaMs_ * 1.05;
            if (next > 10.0 && next < 33.3) targetMs_ = next;
        }
        if ((frameCounter_ % 60u) == 0u) {
            LUCHS_LOG_HOST("[FrameBudget] measured=%.3f ms target=%.3f ms ema=%.3f ms adaptive=%d",
                           lastMeasuredMs_, targetMs_, emaMs_, adaptive_ ? 1 : 0);
        }
    }

    void sleepToCap() const {
        double remaining = targetMs_ - lastMeasuredMs_;
        if (remaining <= 0.0) return;
        using clock = std::chrono::steady_clock;
        auto start = clock::now();
        auto coarse = std::chrono::milliseconds(static_cast<int>(remaining));
        if (coarse.count() > 0) std::this_thread::sleep_for(coarse);
        for (;;) {
            auto now = clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - start).count();
            if (elapsed >= remaining) break;
            std::this_thread::yield();
        }
    }

    double currentTargetMs() const noexcept { return targetMs_; }

private:
    mutable double targetMs_ = 16.666;
    mutable double lastMeasuredMs_ = 0.0;
    mutable double emaMs_ = 0.0;
    mutable unsigned frameCounter_ = 0u;
    bool adaptive_ = false;
};

} // namespace otterdream
