///// Otter: Settings - Axolotel-HUD defaults (center-bottom anchor) + Zoom-Coupler tunables.
///// Schneefuchs: Documented ranges; monotonic clamps in code usage; ASCII-only comments; shader limit synced.
///// Maus: WOW-safe defaults; breath + ring feel; coupler gain modest; perf logs optional.
///// Datei: src/settings_axolotel.hpp
#pragma once

namespace Settings {
namespace Axolotel {

    // Master switch for the additive Axolotel overlay.
    // Range: {false,true} — Default: true
    static constexpr bool enabled = true;

    // === Placement (center bottom) ==========================================
    // Horizontal anchor as fraction of width (0..1). 0.5 = centered.
    // Range: 0.0..1.0 — Default: 0.50
    static constexpr float anchorFracX = 0.50f;

    // Vertical anchor distance from bottom as fraction of height.
    // The runtime uses max(marginBottomPx, anchorFracBottom * height).
    // Range: 0.00..0.50 — Default: 0.08 (8% of height)
    static constexpr float anchorFracBottom = 0.08f;

    // Hard bottom margin in pixels (dominates if larger than fractional distance).
    // Range: 0..512 px — Default: 64 px
    static constexpr float marginBottomPx = 64.0f;

    // === Visuals =============================================================
    // Maximum number of simultaneous pulses the shader will consider.
    // NOTE: Must not exceed the shader uniform array size (uPulses[16]).
    // Range: 1..16 — Default: 8
    static constexpr int   maxPulses = 8;

    // Pulse lifetime in milliseconds (visibility window).
    // Range: 200..2000 ms — Default: 1200 ms
    static constexpr float pulseMs = 1200.0f;

    // Base core glow radius around the anchor (pixels).
    // Range: 0..256 px — Default: 36 px
    static constexpr float coreRadiusPx = 36.0f;

    // Base core glow alpha (before breathing), dimensionless 0..1.
    // Range: 0.00..1.00 — Default: 0.35
    static constexpr float coreAlpha = 0.35f;

    // Breathing modulation amplitude for the core glow (adds to coreAlpha).
    // Range: 0.00..0.40 — Default: 0.15
    static constexpr float breathAmp = 0.15f;

    // Breathing frequency in Hertz (cycles per second).
    // Range: 0.05..3.0 Hz — Default: 0.35 Hz
    static constexpr float breathHz  = 0.35f;

    // Ring width in pixels (thickness of the bright band).
    // Range: 4..64 px — Default: 24 px
    static constexpr float ringWidthPx = 24.0f;

    // Ring radial growth speed in pixels per second.
    // Range: 80..800 px/s — Default: 320 px/s
    static constexpr float ringGrowPxPerSec = 320.0f;

    // Initial ring radius in pixels (at t=0).
    // Range: 0..128 px — Default: 30 px
    static constexpr float ringRadius0Px = 30.0f;

    // Hue base and swing (HSV), both in 0..1.
    // Defaults: base=0.80, swing=0.10
    static constexpr float hueBase  = 0.80f;
    static constexpr float hueSwing = 0.10f;

    // === Logging / Diagnostics ===============================================
    // Overlay performance/activity logging (ASCII single-line).
    // Range: {false,true} — Default: true
    static constexpr bool perfLog = true;

    // === Group colors (kept for future use) ==================================
    static constexpr float colorNav[3]     = { 0.13f, 0.70f, 0.65f };
    static constexpr float colorOverlay[3] = { 0.88f, 0.75f, 0.20f };
    static constexpr float colorSystem[3]  = { 0.85f, 0.30f, 0.70f };
    static constexpr float colorOther[3]   = { 0.25f, 0.60f, 0.95f };

    // ========================================================================
    // Axolotel -> Zoom Coupler (activity energy -> zoom boost)
    // ========================================================================

    // Enable the zoom coupler (maps energy to multiplicative boost).
    // Range: {false,true} — Default: true
    static constexpr bool  couplerEnabled = true;

    // Max gain added on top of 1.0. Example: 0.25 -> boost in [1.00 .. 1.25]
    // Range: 0.00..1.00 — Default: 0.25
    static constexpr float couplerGain    = 0.25f;

    // Attack (rise) time constant in milliseconds (larger = slower ramp-up).
    // Range: 10..1000 ms — Default: 120 ms
    static constexpr float couplerRiseMs  = 120.0f;

    // Decay (release) time constant in milliseconds (larger = slower fall-down).
    // Range: 10..2000 ms — Default: 380 ms
    static constexpr float couplerDecayMs = 380.0f;

    // Per-frame diagnostic logging (ASCII one-liners).
    // Range: {false,true} — Default: false
    static constexpr bool  couplerPerfLog = false;

} // namespace Axolotel
} // namespace Settings
