// LUCHS
#pragma once
// 🦦 Otter: Simple, deterministic estimator for "max FPS" based on core GPU time (uncapped). (Bezug zu Otter)
// 🦊 Schneefuchs: Header/Source strikt getrennt, API minimal, keine ABI-Überraschungen. (Bezug zu Schneefuchs)
// Logs/Strings: ASCII only.

namespace FpsMeter {

/// Update once per frame with the *core* compute time in milliseconds
/// (e.g., mandelbrotMs + entropyMs + contrastMs). Thread-safe.
void updateCoreMs(double coreMs);

/// Get smoothed "uncapped" max FPS as integer for HUD (rounded).
[[nodiscard]] int currentMaxFpsInt();

/// Same as above but as double (no rounding).
[[nodiscard]] double currentMaxFps();

/// Reset internal EMA state (e.g., on major resolution change).
void reset();

} // namespace FpsMeter
