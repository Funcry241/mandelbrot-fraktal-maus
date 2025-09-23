///// Otter: Capybara settings (documented) — drop-in include for your existing settings.hpp.
///// Schneefuchs: ASCII-only; inline constexpr; safe defaults; compile-time sanity checks.
///// Maus: Add this line in settings.hpp (near other toggles):  #include "settings_capybara.inc.hpp"
///// Datei: src/settings_capybara.inc.hpp

#pragma once

// --------------------------------------------------------------------------------------
// Purpose
// --------------------------------------------------------------------------------------
// This file defines **project-level** Capybara knobs inside `namespace Settings` so that
// `capybara_settings.cuh` can map its CAPY_* macros via CAPY_USE_PROJECT_SETTINGS.
// Include this from your existing `settings.hpp` (one line) to activate the bridge.

// --------------------------------------------------------------------------------------
// Settings — documented defaults
// --------------------------------------------------------------------------------------
namespace Settings {

    // ----------------------------------------------------------------------------------
    // capybaraEnabled
    // ----------------------------------------------------------------------------------
    // Purpose: Master on/off switch for the Capybara early Hi/Lo phase.
    // Type   : bool
    // Default: true
    // Range  : {false, true}
    // ↑ Incr : (n/a)
    // ↓ Decr : (n/a)
    // Effect : true  → Use Hi/Lo for the first CAPY_EARLY_ITERS iterations, then classic double.
    //          false → Classic double-only path (no Capybara code executed).
    inline constexpr bool capybaraEnabled = true;

    // ----------------------------------------------------------------------------------
    // capybaraHiLoEarlyIters
    // ----------------------------------------------------------------------------------
    // Purpose: Iteration budget for the early Hi/Lo phase before handing off to classic double.
    // Type   : int
    // Default: 64
    // Range  : [0, 1024]    (recommended operational window)
    // ↑ Incr : Extends the compensated phase → more accuracy at extreme zooms, slightly more cost.
    // ↓ Decr : Shrinks/Disables early phase (0 disables) → faster, but less headroom vs. cancellation.
    // Notes  : 64 is a sweet spot for deep zoom stability without noticeable overhead on SM80+.
    inline constexpr int capybaraHiLoEarlyIters = 64;

    // ----------------------------------------------------------------------------------
    // capybaraRenormRatio
    // ----------------------------------------------------------------------------------
    // Purpose: Renormalization trigger when |lo| grows "too large" compared to |hi|.
    // Type   : double
    // Default: 2^-48 ≈ 3.5527136788005009e-15
    // Range  : (0, 1e-10]  (typical); use smaller values for stricter folding, larger for laxer behavior.
    // ↑ Incr : Less frequent folding (lo allowed to grow more) → marginally faster, slightly riskier numerics.
    // ↓ Decr : More frequent folding → numerically safer, tiny added cost.
    // Effect : When |lo| > capybaraRenormRatio * |hi|, we fold lo back into hi via quick-two-sum.
    inline constexpr double capybaraRenormRatio = 3.5527136788005009e-15;

    // ----------------------------------------------------------------------------------
    // capybaraMappingExactStep
    // ----------------------------------------------------------------------------------
    // Purpose: Use frexp/ldexp to split pixel step into mantissa·2^exp for deep-zoom mapping hygiene.
    // Type   : bool
    // Default: true
    // Range  : {false, true}
    // ↑ Incr : (n/a)
    // ↓ Decr : false → simple multiply (slightly faster), but can lose structure at extreme zoom.
    // Effect : true keeps step structure stable (helps with leading-zero cancellation).
    inline constexpr bool capybaraMappingExactStep = true;

    // ----------------------------------------------------------------------------------
    // capybaraDebugLogging
    // ----------------------------------------------------------------------------------
    // Purpose: Enable device-side ASCII telemetry (rate-limited) for Capybara (init/step/renorm logs).
    // Type   : bool
    // Default: false
    // Range  : {false, true}
    // ↑ Incr : Enables logs → helpful for diagnosis; ensure device log buffer is large enough.
    // ↓ Decr : Disable logs for max performance.
    // Notes  : When true, set a reasonable capybaraLogRate to avoid flooding.
    inline constexpr bool capybaraDebugLogging = false;

    // ----------------------------------------------------------------------------------
    // capybaraLogRate
    // ----------------------------------------------------------------------------------
    // Purpose: Rate limiter for device logs (log every Nth matching event).
    // Type   : int
    // Default: 8192
    // Range  : [0, 1'000'000]    (0 disables logging even if capybaraDebugLogging=true)
    // ↑ Incr : Fewer log lines (sparser) → lower overhead.
    // ↓ Decr : More log lines (denser) → higher overhead; beware of buffer capacity.
    inline constexpr int capybaraLogRate = 8192;

} // namespace Settings

// --------------------------------------------------------------------------------------
// Compile-time sanity checks (guard against accidental misconfiguration)
// --------------------------------------------------------------------------------------
static_assert(Settings::capybaraHiLoEarlyIters >= 0,      "capybaraHiLoEarlyIters must be >= 0");
static_assert(Settings::capybaraRenormRatio > 0.0,        "capybaraRenormRatio must be > 0");
static_assert(Settings::capybaraLogRate >= 0,             "capybaraLogRate must be >= 0");

// --------------------------------------------------------------------------------------
// Integration note:
// In your build, define CAPY_USE_PROJECT_SETTINGS=1 (globally or in the TU that includes
// capybara_prelude.cuh) so that Capybara headers consume these Settings constants.
// Example (CMake):
//   target_compile_definitions(<target> PRIVATE CAPY_USE_PROJECT_SETTINGS=1)
//
// That’s it — the Capybara components will now honor these project-level settings.
// --------------------------------------------------------------------------------------
