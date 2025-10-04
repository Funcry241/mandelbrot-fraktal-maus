///// Otter: Central config; every value documented (purpose, range, default).
///// Schneefuchs: No hidden macros; single source of truth for flags.
///// Maus: performanceLogging=1, ForceAlwaysZoom=1 baseline; ASCII-only logs.
///// Datei: src/settings.hpp

#pragma once

// ============================================================================
// Central project settings – fully documented (only active/used switches).
// Policy: All runtime LOG/DEBUG output must be English and ASCII-only.
// No hidden semantics. Values are stable and compile-time constant.
// ============================================================================

namespace Settings {

// ============================== Zoom / Planner ===============================

    // Forces continuous zoom, independent of entropy/contrast signals.
    // Range: {false, true} | Default: true
    inline constexpr bool   ForceAlwaysZoom = true;

    // After (re-)targeting, freeze direction for this time for stability.
    // Range: 0.2 .. 2.0 (seconds) | Default: 1.0
    inline constexpr double warmUpFreezeSeconds = 1.0;

// ============================== Logging / Perf ===============================

    // Targeted debug/diagnostic output (host/device).
    // Range: {false, true} | Default: false
    inline constexpr bool debugLogging  = false;

    // Condensed [PERF] logs along the frame pipeline.
    // Range: {false, true} | Default: true
    inline constexpr bool performanceLogging = false;

    // --- ZoomLog --------------------------------------------------------------
    // Foundation telemetry for all zoom stages (S1..Sn). Compact, ASCII-only.
    // One optional header line plus a rate-limited data line.
    namespace ZoomLog {
        inline constexpr bool enabled       = true;  // {false,true}
        inline constexpr int  everyN        = 16;    // 1..120
        inline constexpr bool header        = true;  // {false,true}
        inline constexpr bool includeCenter = true;  // {false,true}
    } // namespace ZoomLog

// ============================== Framerate / VSync ============================

    inline constexpr bool capFramerate = true; // {false,true}
    inline constexpr int  capTargetFps = 60;   // 30..240
    inline constexpr bool preferVSync  = true; // {false,true}

// ============================== Interop / Upload =============================

    // Number of PBOs in the CUDA<->OpenGL streaming ring.
    // Typical: 3..12 | Sweet-spot: 4..8 | Default: 8
    inline constexpr int pboRingSize = 8;

// ============================== Overlays / HUD ===============================

    inline constexpr bool  heatmapOverlayEnabled        = true;   // {false,true}
    inline constexpr bool  warzenschweinOverlayEnabled  = true;   // {false,true}
    inline constexpr float hudPixelSize                 = 0.0025f;// 0.0015..0.004

// ============================== Start / Window ===============================

    inline constexpr int   width      = 1024;  // px
    inline constexpr int   height     = 768;   // px
    inline constexpr int   windowPosX = 100;   // px
    inline constexpr int   windowPosY = 100;   // px

    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = 0.0f;
    inline constexpr float initialOffsetY = 0.0f;

// ============================== Iterations / Tiles ===========================

    // Start budget (per-pixel iterations). Grows dynamically.
    // Range: 50 .. 400 | Default: 100
    inline constexpr int INITIAL_ITERATIONS = 100;

    // Hard upper bound for iterations/pixel (safety).
    // Range: 10'000 .. 200'000 | Default: 50'000
    inline constexpr int MAX_ITERATIONS_CAP = 50000;

    // CUDA kernel tiling (constraint: MIN ≤ BASE ≤ MAX).
    // Default: 32 / 8 / 64
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Mandelbrot Kernel ============================
// Used by colorizer and other CUDA launches for thread block geometry.
    inline constexpr int MANDEL_BLOCK_X = 32;   // threads in X (multiple of 32)
    inline constexpr int MANDEL_BLOCK_Y = 8;    // threads in Y

// ============================== Progressive / State ==========================

    // Progressive state toggle (used by RendererState).
    inline constexpr bool progressiveEnabled = true;

// ============================== Kolibri/Grid =================================
// Screen-constant analysis grid (independent of zoom).
// frame_pipeline computes tileSizePx from window size.
// -----------------------------------------------------------------------------
namespace Kolibri {
    inline constexpr bool gridScreenConstant = true; // {false,true}
    inline constexpr int  desiredTilePx      = 28;   // 20..40
} // namespace Kolibri

// ============================== Target Bias ==================================
// Prefer nearer/farther targets on screen (overlay scoring).
// score_biased = raw * ((1 - |mix|) + |mix| * w), with
//   w = exp(-r_ndc^2 / sigmaNdc^2)      if mix >= 0 (center bias)
//   w = 1 - exp(-r_ndc^2 / sigmaNdc^2)  if mix <  0 (edge  bias)
// -----------------------------------------------------------------------------
namespace TargetBias {
    inline constexpr bool   enabled  =  true;   // {false,true}
    inline constexpr double sigmaNdc =  0.55;   // 0.3..1.2
    inline constexpr double mix      = -0.45;   // [-1..1] (negativ = Rand bevorzugt)
} // namespace TargetBias

// ============================== Sanity checks ================================

static_assert(pboRingSize > 0, "pboRingSize must be > 0");
static_assert(MIN_TILE_SIZE <= BASE_TILE_SIZE && BASE_TILE_SIZE <= MAX_TILE_SIZE,
              "MIN_TILE_SIZE <= BASE_TILE_SIZE <= MAX_TILE_SIZE required");
static_assert(Kolibri::desiredTilePx > 0, "desiredTilePx must be > 0");
static_assert(MANDEL_BLOCK_X > 0 && MANDEL_BLOCK_Y > 0, "MANDEL_BLOCK dims must be > 0");
static_assert(TargetBias::sigmaNdc > 0.0, "sigmaNdc must be > 0");
static_assert(TargetBias::mix >= -1.0 && TargetBias::mix <= 1.0, "mix in [-1,1]");

} // namespace Settings
