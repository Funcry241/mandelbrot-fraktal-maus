///// src/frame_limiter.hpp
// ============================================================================
// Datei: src/frame_limiter.hpp
// FrameLimiter — precise 60 FPS pacing (sleep+spin), ASCII logs only.
// 🐭 Maus: deterministisch, jitterarm, header-only, keine versteckten Abhängigkeiten.
// 🦦 Otter: sanftes Pacing mit fein granularer Spin-Phase für butterweiche Bewegung. (Bezug zu Otter)
// 🦊 Schneefuchs: Kein Zeitkriechen — Driftkorrektur bei Overruns; Logs optional. (Bezug zu Schneefuchs)
// ============================================================================

#pragma once

#include <chrono>
#include <thread>
#include <cmath>            // für std::abs
#include "settings.hpp"
#include "luchs_log_host.hpp"

namespace pace { // umbenannt von 'fps' → keine Makro-Kollisionen

class FrameLimiter {
public:
    FrameLimiter() = default;

    // Limit the calling thread to targetFps. If targetFps <= 0, does nothing.
    inline void limit(int targetFps) {
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();

        if (targetFps <= 0) {
            if (_initialized) {
                _lastDt = std::chrono::duration<double>(now - _lastStamp).count();
            }
            _lastStamp = now;
            _initialized = true;
            return;
        }

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / static_cast<double>(targetFps)));

        if (!_initialized) {
            _initialized = true;
            _nextTick    = now + period; // first frame: no sleep
            _lastStamp   = now;
            _lastDt      = 0.0;
            _lastSleep   = 0.0;
            _logCounter  = 0;
            return;
        }

        // Coarse sleep if far from _nextTick, then fine spin-wait.
        auto remaining = _nextTick - now;
        double sleptMs = 0.0;

        if (remaining > std::chrono::milliseconds(2)) {
            const auto coarse = remaining - std::chrono::milliseconds(1);
            std::this_thread::sleep_for(coarse);
            sleptMs += std::chrono::duration<double, std::milli>(coarse).count();
        }

        // Fine spin — keeps jitter low without timeBeginPeriod.
        for (;;) {
            const auto n = clock::now();
            if (n >= _nextTick) break;
            std::this_thread::yield();
        }

        const auto after = clock::now();
        _lastSleep = sleptMs;
        _lastDt    = std::chrono::duration<double>(after - _lastStamp).count();
        _lastStamp = after;

        // Drift handling: if we overran significantly, rebase to avoid creep.
        if (after - _nextTick > period) {
            _nextTick = after + period; // re-sync (no accumulated lag)
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[FPS] overrun: frame=%.3fms; re-sync next tick", _lastDt * 1000.0);
            }
        } else {
            _nextTick += period;
        }

        // Sparse pacing log.
        if constexpr (Settings::performanceLogging) {
            if (++_logCounter >= 120) {
                _logCounter = 0;
                const double targetMs = 1000.0 / static_cast<double>(targetFps);
                const double jitterMs = std::abs(_lastDt * 1000.0 - targetMs);
                LUCHS_LOG_HOST("[FPS] target=%d dt=%.3fms sleep=%.3fms jitter=%.3fms",
                               targetFps, _lastDt * 1000.0, _lastSleep, jitterMs);
            }
        }
    }

    inline double lastFrameSeconds() const { return _lastDt; }
    inline double lastSleepMilliseconds() const { return _lastSleep; }

    inline void reset() {
        _initialized = false;
        _logCounter  = 0;
        _lastDt      = 0.0;
        _lastSleep   = 0.0;
    }

private:
    std::chrono::steady_clock::time_point _nextTick{};
    std::chrono::steady_clock::time_point _lastStamp{};
    bool   _initialized{false};
    int    _logCounter{0};
    double _lastDt{0.0};
    double _lastSleep{0.0};
};

} // namespace pace
