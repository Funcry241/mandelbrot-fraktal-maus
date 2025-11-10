///// Otter: Central config – Nacktmull + Replikatoren; alle Werte dokumentiert; deterministische ASCII-Logs.
/***** Schneefuchs: Keine versteckten Makros; Header/Source synchron; /WX-safe Defaults; GLEW dynamisch. *****/
/***** Maus: performanceLogging=1, ForceAlwaysZoom=1; Kolibri-Grid; AiBandit-Block (Shadow/Assisted/Auto Ready). *****/
///// Datei: src/settings.hpp
#pragma once

// ============================================================================
// Central project settings – only active switches live here.
// All runtime logs must be English, ASCII-only.
// Tip: Prefer small, deliberate changes and measure. Keep headers/sources in sync.
// ============================================================================

namespace Settings {

// ============================== Zoom / Planner ===============================
    inline constexpr bool   ForceAlwaysZoom      = true;
    inline constexpr double warmUpFreezeSeconds  = 1.0;

// ============================== Logging / Perf ===============================
    inline constexpr bool debugLogging           = false;
    inline constexpr bool performanceLogging     = true;

    namespace ZoomLog {
        inline constexpr bool enabled       = true;
        inline constexpr int  everyN        = 16;
        inline constexpr bool header        = true;
        inline constexpr bool includeCenter = true;
    }

    namespace PerfLog {
        inline constexpr bool enabled      = true;
        inline constexpr int  everyN       = 1;
        inline constexpr int  warmupFrames = 0;
        inline constexpr bool header       = true;
        inline constexpr bool emitCudaLine = false;
        inline constexpr bool compact      = false;
    }

// ============================== Framerate / VSync ============================
    inline constexpr bool capFramerate = true;
    inline constexpr int  capTargetFps = 60;
    inline constexpr bool preferVSync  = true;

// ============================== Interop / Upload =============================
    inline constexpr int pboRingSize = 8;

// ============================== Overlays / HUD ===============================
    inline constexpr bool  heatmapOverlayEnabled       = true;
    inline constexpr bool  warzenschweinOverlayEnabled = true;
    inline constexpr float hudPixelSize                = 0.0025f;

// ============================== Start / Window ===============================
    inline constexpr int   width      = 1024;
    inline constexpr int   height     = 768;
    inline constexpr int   windowPosX = 100;
    inline constexpr int   windowPosY = 100;
    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = 0.0f;
    inline constexpr float initialOffsetY = 0.0f;

// ============================== Iterations / Tiles ===========================
    inline constexpr int INITIAL_ITERATIONS = 100;
    inline constexpr int MAX_ITERATIONS_CAP = 50000;
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Mandelbrot Kernel ============================
    inline constexpr int MANDEL_BLOCK_X = 32;
    inline constexpr int MANDEL_BLOCK_Y = 8;

// ============================== Progressive / State ==========================
    inline constexpr bool progressiveEnabled = true;

// ============================== Kolibri / Grid ===============================
namespace Kolibri {
    inline constexpr bool gridScreenConstant = true;
    inline constexpr int  desiredTilePx      = 28;
}

// ============================== Stats Cadence ================================
namespace StatsCadence {
    inline constexpr int heatmapEveryN = 3;
}

// ============================== Target Bias ==================================
namespace TargetBias {
    inline constexpr bool   enabled  = true;
    inline constexpr double sigmaNdc = 0.65;
    inline constexpr double mix      = 0.35;
}

// ============================== Nav Bias (WASD/Arrows) =======================
namespace NavBias {
    inline constexpr bool   enabled     = true;
    inline constexpr double gainPerSec  = 1.6;
    inline constexpr double halfLifeSec = 0.8;
    inline constexpr double maxNdc      = 0.28;
    inline constexpr double yScale      = 0.94;
}

// ============================== Sanity checks ================================
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
static_assert(NavBias::gainPerSec  >= 0.0, "gainPerSec must be >= 0");
static_assert(NavBias::halfLifeSec >  0.0, "halfLifeSec must be > 0");
static_assert(NavBias::maxNdc      >= 0.0, "maxNdc must be >= 0");

// ============================== Replikatoren ================================
namespace Perturb {
    inline constexpr bool   enabled       = true;   // Ctrl+P toggles runtime
    inline constexpr int    gatePixelSize = 12;
    inline constexpr double deltaScale    = 1.0;
    inline constexpr int    sandboxTile   = -1;     // -1 off
}

namespace Ai {
    inline constexpr bool        enabled    = true;
    inline constexpr bool        aopEnabled = true;
    inline constexpr const char* ep         = "cuda";
    inline constexpr float wE = 0.60f;
    inline constexpr float wC = 0.40f;
}

// *** Neu: selbstlernender Bandit (LinUCB/RLS), komplett on-device ***********
// Stage: 0=Shadow (lernt, entscheidet nicht), 1=Assisted (Kandidaten), 2=Auto (steuert Ziel)
namespace AiBandit {
    inline constexpr int   stage            = 0;        // Start sicher im Shadow
    inline constexpr int   topK             = 3;        // Kandidaten pro Takt
    inline constexpr int   retargetInterval = 5;        // Frames pro Update
    inline constexpr float alpha            = 0.8f;     // UCB-Exploration
    inline constexpr float epsilon          = 0.05f;    // ε-Exploration
    inline constexpr float lambda           = 1.0e-2f;  // RLS-Regularisierung
    inline constexpr float beta             = 0.6f;     // Reward: E + β·C
    inline constexpr float rewardClampLo    = -1.0f;    // Reward-Clamps
    inline constexpr float rewardClampHi    = +1.0f;
    inline constexpr unsigned int seed      = 0xC0FFEEu;// deterministische RNG-Quelle
    inline constexpr const char* persistPath = "dist/ai/otter_bandit.bin"; // optional
}

} // namespace Settings
