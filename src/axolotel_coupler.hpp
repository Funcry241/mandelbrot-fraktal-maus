///// Otter: Axolotel Zoom-Coupler — maps key-pulse energy to a smooth zoom boost (attack/release, WOW but safe).
///// Schneefuchs: Header-only API; deterministic clamps; ASCII logs; no GL deps; stable ABI surface.
///// Maus: get boost() in [1..1+gain]; tick(dt) per frame; optional perf log; Settings live in settings_axolotel.hpp.
///// Datei: src/axolotel_coupler.hpp
#pragma once

namespace AxolotelCoupler {

// Lifetime
void init();
void shutdown();

// Per-frame update (dtSeconds >= 0)
void tick(float dtSeconds);

// Query current smoothed energy (0..1) and multiplicative boost (>=1)
float energy();              // smoothed E in [0,1]
float boost();               // multiplier in [1, 1 + gain]

// Enable/disable at runtime (soft switch; preserves state)
void setEnabled(bool enabled);
bool isEnabled();

} // namespace AxolotelCoupler
