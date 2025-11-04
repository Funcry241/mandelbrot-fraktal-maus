///// Otter: Central config – Nacktmull defaults consolidated; every value documented (purpose, range, default)
///// Schneefuchs: No hidden macros; single source of truth for flags & cadences; ASCII-only policy
///// Maus: performanceLogging=1, ForceAlwaysZoom=1 baseline; 32×8 blocks; no fast-math; deterministic logs
///// Datei: src/settings.hpp

#pragma once

// ============================================================================
// Central project settings – only active switches live here.
// All runtime logs must be English, ASCII-only.
// Tip: Prefer small, deliberate changes and measure. Keep headers/sources in sync.
// ============================================================================

namespace Settings {

// ============================== Zoom / Planner ===============================
// Controls the auto-zoom planner’s global behavior.

    // Force continuous zoom regardless of entropy/contrast signals.
    // Use when exploring or for demos to avoid stalls.
    // Range: {false, true} | Default: true
    inline constexpr bool   ForceAlwaysZoom = true;

    // After a new target is chosen, keep direction fixed for stability.
    // Higher values = fewer direction flips but slower reaction.
    // Range: 0.2 .. 2.0 seconds | Default: 1.0
    inline constexpr double warmUpFreezeSeconds = 1.0;

// ============================== Logging / Perf ===============================
// Toggle targeted debug and compact perf logs. These do not change math;
// they only affect I/O and timing jitter from printing.

    // Verbose diagnostics (host/device). Keep off for clean benchmarks.
    // Range: {false, true} | Default: false
    inline constexpr bool debugLogging  = false;

    // Compact [PERF] lines along the frame pipeline (timings, FPS, ring).
    // Range: {false, true} | Default: true
    inline constexpr bool performanceLogging = true;

    // --- ZoomLog --------------------------------------------------------------
    // Telemetry for zoom stages (S1..Sn). Rate-limited, optional header.
    namespace ZoomLog {
        // Emit zoom telemetry lines.
        // Range: {false, true} | Default: true
        inline constexpr bool enabled       = true;

        // Emit every Nth frame (1 = every frame). Larger N reduces log noise.
        // Range: 1 .. 120 | Default: 16
        inline constexpr int  everyN        = 16;

        // Print a single header explaining columns on first emission.
        // Range: {false, true} | Default: true
        inline constexpr bool header        = true;

        // Include current complex center in the log line (useful for replay).
        // Range: {false, true} | Default: true
        inline constexpr bool includeCenter = true;
    } // namespace ZoomLog

    // --- PerfLog cadence (Nacktmull) -----------------------------------------
    // Rate-limits hot-path [PERF] lines and defines a warm-up window.
    // Does not affect computation; reduces I/O variance.
    namespace PerfLog {
        // Enable [PERF] lines.
        // Range: {false, true} | Default: true
        inline constexpr bool enabled      = true;

        // Emit every Nth frame after warm-up (1 = every frame).
        // Range: 10 .. 240 | Default: 20
        inline constexpr int  everyN       = 20;

        // Suppress perf logs during first frames to avoid cold-start noise.
        // Range: 0 .. 300 | Default: 60
        inline constexpr int  warmupFrames = 60;

        // Emit a single header explaining columns on first emission.
        // Range: {false, true} | Default: true
        inline constexpr bool header       = true;
    } // namespace PerfLog

// ============================== Framerate / VSync ============================
// Frame pacing. Prefer VSync for visual stability; cap for headroom.

    // Hard cap in the main loop. Keep <= monitor refresh when preferVSync=true.
    // Range: {false, true} | Default: true
    inline constexpr bool capFramerate = true;

    // Target FPS when capFramerate=true.
    // Range: 30 .. 240 | Default: 60
    inline constexpr int  capTargetFps = 60;

    // Ask GL for VSync; driver may override. Turn off for raw perf tests.
    // Range: {false, true} | Default: true
    inline constexpr bool preferVSync  = true;

// ============================== Interop / Upload =============================
// PBO ring for GL upload. Larger rings reduce stalls but use more VRAM.

    // Number of PBOs in the ring buffer.
    // Range: 3 .. 12 | Default: 8
    inline constexpr int pboRingSize = 8;

// ============================== Overlays / HUD ===============================
// Visual diagnostics on top of the fractal output.

    // Heatmap overlay (entropy/contrast tiles).
    // Range: {false, true} | Default: true
    inline constexpr bool  heatmapOverlayEnabled       = true;

    // Warzenschwein HUD text (stats + status).
    // Range: {false, true} | Default: true
    inline constexpr bool  warzenschweinOverlayEnabled = true;

    // Text size in NDC; larger = bigger glyphs.
    // Range: 0.0015 .. 0.004 | Default: 0.0025
    inline constexpr float hudPixelSize                = 0.0025f;

// ============================== Start / Window ===============================
// Initial window size/position and view parameters.

    // Window resolution (pixels).
    inline constexpr int   width      = 1024;
    inline constexpr int   height     = 768;

    // Initial window position (pixels).
    inline constexpr int   windowPosX = 100;
    inline constexpr int   windowPosY = 100;

    // Initial view in complex plane.
    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = 0.0f;
    inline constexpr float initialOffsetY = 0.0f;

// ============================== Iterations / Tiles ===========================
// Iteration budget and compute tile size clamps.

    // Starting iteration budget; may ramp with zoom.
    // Range: 50 .. 400 | Default: 100
    inline constexpr int INITIAL_ITERATIONS = 100;

    // Absolute ceiling for iteration budget (safety).
    // Range: 10000 .. 200000 | Default: 50000
    inline constexpr int MAX_ITERATIONS_CAP = 50000;

    // Tile size baseline and clamps for compute kernels.
    // Constraint: MIN <= BASE <= MAX
    // Typical: 8..64 depending on zoom and occupancy.
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Mandelbrot Kernel ============================
// Thread block geometry used by colorizer/metrics (render TU may override
// via __launch_bounds__). Keep X a multiple of 32 for warp alignment.

    // Threads in X (must be multiple of 32).
    inline constexpr int MANDEL_BLOCK_X = 32;

    // Threads in Y.
    inline constexpr int MANDEL_BLOCK_Y = 8;

    // Note: The render kernel translation unit can set its own launch_bounds
    // for occupancy. MANDEL_BLOCK_* is the shared default for other launches.

// ============================== Progressive / State ==========================
// Persistent state across frames (resume iterations, etc.).

    // Toggle progressive renderer features in RendererState.
    // Range: {false, true} | Default: true
    inline constexpr bool progressiveEnabled = true;

// ============================== Kolibri / Grid ===============================
// Screen-constant analysis grid independent of zoom. The frame pipeline
// picks tile size in pixels from window size for overlays/metrics.

namespace Kolibri {
    // Keep analysis grid constant in screen space (pixels) instead of world space.
    // Range: {false, true} | Default: true
    inline constexpr bool gridScreenConstant = true;

    // Desired tile size for the screen-constant grid (pixels).
    // Range: 20 .. 40 | Default: 28
    inline constexpr int  desiredTilePx      = 28;

    // NOTE: legacy Kolibri::metricsEveryN removed (moved to StatsCadence).
} // namespace Kolibri

// ============================== Stats Cadence ================================
// Rate-limit for analysis metrics (entropy/contrast) to save time without
// changing visuals. Compute metrics only every Nth frame; reuse the last
// results in between.
namespace StatsCadence {
    // Compute heatmap metrics every Nth frame.
    // 1 = every frame; 3 = balanced default; larger = lighter load.
    // Range: 1 .. 16 | Default: 3
    inline constexpr int heatmapEveryN = 3;
} // namespace StatsCadence

// ============================== Target Bias ==================================
// Center-weighted scoring for interest selection in overlays.
// score_biased = raw * ((1 - mix) + mix * exp(-r_ndc^2 / sigmaNdc^2))
// Smaller sigma ⇒ stronger center bias; mix blends raw vs. biased.

namespace TargetBias {
    // Enable bias toward the screen center.
    // Range: {false, true} | Default: true
    inline constexpr bool   enabled  = true;

    // Width of the Gaussian in NDC; smaller = tighter center pull.
    // Range: 0.3 .. 1.2 | Default: 0.65
    inline constexpr double sigmaNdc = 0.65;

    // Blend factor between raw and biased score.
    // Range: 0 .. 1 | Default: 0.35
    inline constexpr double mix      = 0.35;
} // namespace TargetBias

// ============================== Sanity checks ================================
// Guard obvious configuration errors at compile time.

static_assert(pboRingSize > 0, "pboRingSize must be > 0");
static_assert(MIN_TILE_SIZE <= BASE_TILE_SIZE && BASE_TILE_SIZE <= MAX_TILE_SIZE,
              "MIN_TILE_SIZE <= BASE_TILE_SIZE <= MAX_TILE_SIZE required");
static_assert(Kolibri::desiredTilePx > 0, "desiredTilePx must be > 0");
static_assert(StatsCadence::heatmapEveryN >= 1, "StatsCadence::heatmapEveryN must be >= 1");
static_assert(MANDEL_BLOCK_X > 0 && MANDEL_BLOCK_Y > 0, "MANDEL_BLOCK dims must be > 0");
static_assert((MANDEL_BLOCK_X % 32) == 0, "MANDEL_BLOCK_X must be a multiple of 32");
static_assert(TargetBias::sigmaNdc > 0.0, "sigmaNdc must be > 0");
static_assert(TargetBias::mix >= 0.0 && TargetBias::mix <= 1.0, "mix in [0,1]");
static_assert(ZoomLog::everyN >= 1, "ZoomLog::everyN must be >= 1");
static_assert(PerfLog::everyN >= 1, "PerfLog::everyN must be >= 1");
static_assert(PerfLog::warmupFrames >= 0, "PerfLog::warmupFrames must be >= 0");

} // namespace Settings
