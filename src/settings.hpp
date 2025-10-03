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

    // ForceAlwaysZoom
    // Forces continuous zoom, independent of entropy/contrast signals.
    // Range: {false, true} | Default: true
    inline constexpr bool   ForceAlwaysZoom = true;

    // warmUpFreezeSeconds
    // After (re-)targeting, freeze direction for this time for stability.
    // Range: 0.2 .. 2.0 (seconds) | Default: 1.0
    inline constexpr double warmUpFreezeSeconds = 1.0;

// ============================== Logging / Perf ===============================

    // debugLogging
    // Targeted debug/diagnostic output (host/device).
    // Range: {false, true} | Default: true
    inline constexpr bool debugLogging  = false;

    // performanceLogging
    // Condensed [PERF] logs along the frame pipeline.
    // Range: {false, true} | Default: true
    inline constexpr bool performanceLogging = false;

    // --- ZoomLog (NEW) -------------------------------------------------------
    // Foundation telemetry for all zoom stages (S1..Sn). Compact, ASCII-only.
    // One optional header line plus a rate-limited data line.
    namespace ZoomLog {
        // enabled
        // Master switch for zoom telemetry.
        // Range: {false, true} | Default: true
        inline constexpr bool enabled = true;

        // everyN
        // Emit a data line every N frames (1 = every frame).
        // Range: 1 .. 120 | Default: 16
        inline constexpr int  everyN  = 16;

        // header
        // Print a single header line once at startup.
        // Range: {false, true} | Default: true
        inline constexpr bool header  = true;

        // includeCenter
        // Append current center (cx,cy) to each data line.
        // Range: {false, true} | Default: true
        inline constexpr bool includeCenter = true;
    } // namespace ZoomLog

// ============================== Framerate / VSync ============================

    // capFramerate
    // CPU-side framerate limiter (sleep+spin) to a target FPS.
    // Range: {false, true} | Default: true
    inline constexpr bool capFramerate = true;

    // capTargetFps
    // Target framerate for the limiter.
    // Range: 30 .. 240 | Default: 60
    inline constexpr int  capTargetFps = 60;

    // preferVSync
    // Prefer VSync when available (swap-coupled).
    // Range: {false, true} | Default: true
    inline constexpr bool preferVSync = true;

// ============================== Interop / Upload =============================

    // pboRingSize
    // Number of PBOs in the CUDA<->OpenGL streaming ring.
    // Typical: 3..12 | Sweet-spot: 4..8 | Default: 8
    inline constexpr int pboRingSize = 8;

// ============================== Overlays / HUD ===============================

    // heatmapOverlayEnabled
    // Enable heatmap overlay.
    // Range: {false, true} | Default: true
    inline constexpr bool heatmapOverlayEnabled = true;

    // warzenschweinOverlayEnabled
    // Enable warzenschwein HUD overlay.
    // Range: {false, true} | Default: true
    inline constexpr bool warzenschweinOverlayEnabled = true;

    // hudPixelSize
    // Base HUD pixel size in NDC.
    // Range: 0.0015f .. 0.0040f | Default: 0.0025f
    inline constexpr float hudPixelSize = 0.0025f;

// ============================== Start / Window ===============================

    // width / height (initial window size, px)
    // Range: 800x600 .. 3840x2160 | Default: 1024x768
    inline constexpr int width  = 1024;
    inline constexpr int height = 768;

    // windowPosX / windowPosY (initial window position, px)
    // Range: >=0 | Default: (100,100)
    inline constexpr int windowPosX  = 100;
    inline constexpr int windowPosY  = 100;

    // initial view (zoom + offset)
    // Default: 1.5 / (0.0, 0.0)
    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = 0.0f;
    inline constexpr float initialOffsetY = 0.0f;

// ============================== Iterations / Tiles ===========================

    // INITIAL_ITERATIONS
    // Start budget (per-pixel iterations). Grows dynamically.
    // Range: 50 .. 400 | Default: 100
    inline constexpr int INITIAL_ITERATIONS = 100;

    // MAX_ITERATIONS_CAP
    // Hard upper bound for iterations/pixel (safety).
    // Range: 10'000 .. 200'000 | Default: 50'000
    inline constexpr int MAX_ITERATIONS_CAP = 50000;

    // BASE/MIN/MAX_TILE_SIZE (CUDA kernel tiling)
    // Constraint: MIN ≤ BASE ≤ MAX | Default: 32 / 8 / 64
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Periodicity Check ============================

    // periodicityEnabled
    // Kernel-level periodicity probe (early exit).
    // Range: {false, true} | Default: true
    inline constexpr bool periodicityEnabled = true;

    // periodicityCheckInterval
    // Probe interval (iterations).
    // Range: 32 .. 128 | Default: 96
    inline constexpr int periodicityCheckInterval = 96;

    // periodicityEps2
    // Squared distance threshold between z-probes.
    // Range: 1e-16 .. 1e-12 | Default: 1e-14
    inline constexpr double periodicityEps2 = 1e-14;

// ============================== Progressive / State ==========================

    // progressiveEnabled
    // Allocate persistent per-pixel state (Z, it) for progressive iteration.
    // Default: true
    inline constexpr bool progressiveEnabled = true;

    // progressiveAddIter
    // Progressive per-frame iteration budget (resume path).
    // Interactivity: 16..48 | Default: 32
    inline constexpr int  progressiveAddIter = 32;

// ============================== Mandelbrot Kernel ============================

    // Block geometry (affects occupancy & coalescing).
    // Recommendation: (32,8) balanced; (32,16) higher ILP (watch registers).
    inline constexpr int MANDEL_BLOCK_X = 32;   // threads in X (multiple of 32)
    inline constexpr int MANDEL_BLOCK_Y = 8;    // threads in Y

    // Unroll hint for inner iteration loop.
    // Range: 1..8 | Default: 4
    inline constexpr int MANDEL_UNROLL  = 4;

    // Enable fused multiply-add in iteration updates.
    inline constexpr bool MANDEL_USE_FMA = true;

// ============================== Capybara (Hi/Lo early) =======================
// Precision-improved early phase: hi+lo accumulation with optional renorm.
// Telemetry is ASCII-only and rate-limited.
// -----------------------------------------------------------------------------

    // capybaraEnabled
    // Master on/off for Capybara early phase.
    // Range: {false, true} | Default: true
    inline constexpr bool capybaraEnabled = true;

    // capybaraHiLoEarlyIters
    // Iteration budget for early hi+lo before classic double.
    // Range: 32 .. 128 (typical) | Default: 64
    inline constexpr int  capybaraHiLoEarlyIters = 64;

    // capybaraRenormLoRatio
    // Renormalize when |lo| > ratio * |hi|.
    // Range: (0, 1e-10] | Default: 2^-48 ≈ 3.552713678800501e-15
    inline constexpr double capybaraRenormLoRatio = 3.552713678800501e-15; // 2^-48

    // capybaraMappingExactStep
    // Use frexp/ldexp-based exact pixel step mapping (deep-zoom hygiene).
    // Range: {false, true} | Default: true
    inline constexpr bool capybaraMappingExactStep = true;

    // CapyFmaMode
    // Controls FMA usage in Capybara path.
    enum class CapyFmaMode : unsigned char { Auto, ForceOn, ForceOff };
    inline constexpr CapyFmaMode capybaraFmaMode = CapyFmaMode::Auto;

    // capybaraDebugLogging
    // Device-side ASCII telemetry (rate-limited).
    // Range: {false, true} | Default: false
    inline constexpr bool capybaraDebugLogging = false;

    // capybaraLogRate
    // Rate limiter for device logs (log every Nth event).
    // Range: 10 .. 120 (project-specific) | Default: 30
    inline constexpr int  capybaraLogRate = 30;

// ============================== Kolibri/Grid =================================
// Screen-constant analysis grid (independent of zoom).
// frame_pipeline computes tileSizePx from window size.
// -----------------------------------------------------------------------------
namespace Kolibri {
    // gridScreenConstant
    // Keep analysis tiles approximately constant in screen pixels.
    // Range: {false, true} | Default: true
    inline constexpr bool gridScreenConstant = true;

    // desiredTilePx
    // Target tile size in screen pixels.
    // Range: 20 .. 40 | Default: 28
    inline constexpr int  desiredTilePx = 28;

    // gridFadeEnable / gridFadeMinFps / gridFadeZoomStart
    // Optional fade-out of the grid if it would distract.
    // Defaults: true / 35 / 5000.0f
    inline constexpr bool  gridFadeEnable    = true;
    inline constexpr int   gridFadeMinFps    = 35;
    inline constexpr float gridFadeZoomStart = 5000.0f;
} // namespace Kolibri

// ============================== Kolibri/Boost ================================
// Deep-zoom framerate stabilizer: frame budget & runtime addIter clamps.
// -----------------------------------------------------------------------------
namespace KolibriBoost {
    // enable
    // Master switch for runtime addIter controller.
    // Default: true
    inline constexpr bool   enable = true;

    // targetFrameMs
    // Desired frame time (ms); pipeline adjusts addIterRuntime around this.
    // Range: 18.0 .. 25.0 | Default: 22.0
    inline constexpr double targetFrameMs = 22.0;

    // addIterMin / addIterMax / addIterStep (per frame)
    // Defaults: 16 / 48 / 2
    inline constexpr int addIterMin  = 16;
    inline constexpr int addIterMax  = 48;
    inline constexpr int addIterStep = 2;
} // namespace KolibriBoost

// ============================== Sanity checks ================================

static_assert(pboRingSize > 0, "pboRingSize must be > 0");
static_assert(MIN_TILE_SIZE <= BASE_TILE_SIZE && BASE_TILE_SIZE <= MAX_TILE_SIZE,
              "MIN_TILE_SIZE <= BASE_TILE_SIZE <= MAX_TILE_SIZE required");
static_assert(Kolibri::desiredTilePx > 0, "desiredTilePx must be > 0");
static_assert(capybaraHiLoEarlyIters >= 0, "capybaraHiLoEarlyIters must be >= 0");
static_assert(capybaraRenormLoRatio > 0.0, "capybaraRenormLoRatio must be > 0");
static_assert(capybaraLogRate >= 0, "capybaraLogRate must be >= 0");

} // namespace Settings
