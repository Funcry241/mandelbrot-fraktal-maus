///// Otter: FPS-Meter-API – glättet Core-Zeit zu stabiler Max-FPS-Anzeige.
///// Schneefuchs: Header/Source strikt getrennt; ABI stabil; ASCII-only.
///// Maus: Keine iostreams; Thread-safety via Atomics in .cpp; API minimal.
///// Datei: src/fps_meter.hpp

#pragma once

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
