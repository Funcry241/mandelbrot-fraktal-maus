///// Otter: Settings – Axolotel-HUD defaults (center-bottom anchor), crisp docs, WOW-safe defaults.
///// Schneefuchs: Documented ranges; monotonic clamps in code-side usage; ASCII-only comments; shader limit synced.
///// Maus: Self-contained header (included by axolotel_hud.cpp); migration into main settings.hpp optional later.
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
    // The ring expands during this window and fades out towards the end.
    // Range: 200..2000 ms — Default: 1200 ms (matches shader’s 1.20 s window)
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
    // Shader currently uses angular frequency ≈ 2.2 rad/s → ~0.35 Hz.
    // Range: 0.05..3.0 Hz — Default: 0.35 Hz (documented; shader uses 2.2 rad/s)
    static constexpr float breathHz  = 0.35f;

    // Ring width in pixels (thickness of the bright band).
    // Range: 4..64 px — Default: 24 px
    static constexpr float ringWidthPx = 24.0f;

    // Ring radial growth speed in pixels per second.
    // Shader uses r = r0 + growPxPerSec * t with r0 ≈ 30 px.
    // Range: 80..800 px/s — Default: 320 px/s
    static constexpr float ringGrowPxPerSec = 320.0f;

    // Initial ring radius in pixels (at t=0).
    // Range: 0..128 px — Default: 30 px
    static constexpr float ringRadius0Px = 30.0f;

    // Hue base and swing for the gradient (HSV space), both in 0..1.
    // The shader uses a gentle hue oscillation over time around hueBase.
    // Range: base 0..1, swing 0..0.5 — Defaults: base=0.80, swing=0.10
    static constexpr float hueBase  = 0.80f;
    static constexpr float hueSwing = 0.10f;

    // === Logging / Diagnostics ===============================================
    // Overlay performance/activity logging (ASCII single-line).
    // Range: {false,true} — Default: true
    static constexpr bool perfLog = true;

    // === Optional group colors (future use) ==================================
    // Not consumed by the current shader; reserved for key-group tinting.
    // Values are linear RGB in 0..1.
    static constexpr float colorNav[3]     = { 0.13f, 0.70f, 0.65f }; // navigation keys
    static constexpr float colorOverlay[3] = { 0.88f, 0.75f, 0.20f }; // HUD/overlay toggles
    static constexpr float colorSystem[3]  = { 0.85f, 0.30f, 0.70f }; // system keys (Esc/F-keys)
    static constexpr float colorOther[3]   = { 0.25f, 0.60f, 0.95f }; // default/other

} // namespace Axolotel
} // namespace Settings
