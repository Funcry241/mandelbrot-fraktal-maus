///// Otter: Zentral-Config – Replikatoren „Replikatoren“ (Bandit Shadow), Nacktmull-Perf; klar dokumentierte Werte.
///// Schneefuchs: /WX-safe; ASCII-Logs; keine versteckten Makros; Header/Source synchron; GLEW dynamisch.
///// Maus: ForceAlwaysZoom=1; PerfLog aktiv; Kolibri-Grid; Luchs-Kompatblock (enabled/nvrtc) wiederhergestellt.
///// Datei: src/settings.hpp
#pragma once

// ============================================================================
// Settings: Alle aktiv schaltbaren Projektparameter an einer Stelle.
// Regeln:
//   • Runtime-Logs ausschließlich EN/ASCII.
//   • Kommentare nennen Zweck, Bereich, Default und Wirkung.
//   • Kleine, gezielte Änderungen → messen → beibehalten/rollen.
//   • Header/Source in Sync halten.
// ============================================================================

namespace Settings {

// ============================== Zoom / Planner ===============================
// Steuert globales Zoomverhalten.
    // Erzwingt kontinuierliches Zoomen – verhindert Stalls in ruhigen Bereichen.
    // Range: {false,true} | Default: true | ↑ stärkerer Demo-Flow, ↓ signalgetriebener
    inline constexpr bool   ForceAlwaysZoom      = true;

    // Fixiert die Bewegungsrichtung nach Zielwechsel für Stabilität.
    // Range: 0.2 .. 2.0 Sekunden | Default: 1.0 | ↑ weniger Richtungswechsel, ↓ reaktiver
    inline constexpr double warmUpFreezeSeconds  = 1.0;

// ============================== Logging / Perf ===============================
// Ein-/Ausgaben beeinflussen nicht die Mathematik, nur I/O-Jitter.
    // Verbose Diagnostik (Host/Device). Für saubere Benchmarks aus.
    // Range: {false,true} | Default: false | ↑ mehr Einblick, ↓ potenziell mehr I/O
    inline constexpr bool debugLogging           = false;

    // Kompakte [PERF]-Zeilen (Timings, FPS, Ring).
    // Range: {false,true} | Default: true | ↑ Telemetrie, ↓ minimales I/O-Rauschen
    inline constexpr bool performanceLogging     = true;

    // Zoom-Telemetry (rate-limitiert).
    namespace ZoomLog {
        // Ausgabe aktivieren.
        // Range: {false,true} | Default: true
        inline constexpr bool enabled       = true;

        // Jede N-te Frame-Zeile (1 = jede).
        // Range: 1 .. 120 | Default: 16 | ↑ weniger Log, ↓ feinere Spur
        inline constexpr int  everyN        = 16;

        // Einmalige Kopfzeile.
        // Range: {false,true} | Default: true
        inline constexpr bool header        = true;

        // Komplexes Zentrum mitschreiben (Replays).
        // Range: {false,true} | Default: true
        inline constexpr bool includeCenter = true;
    }

    // Nacktmull-Perf-Cadence (hot-path [PERF]).
    namespace PerfLog {
        // Ausgabe aktivieren.
        // Range: {false,true} | Default: true
        inline constexpr bool enabled      = true;

        // Jede N-te Frame-Zeile nach Warmup.
        // Range: 1 .. 240 | Default: 1
        inline constexpr int  everyN       = 1;

        // Warmup-Frames ohne Perf-Log (Cold-Start-Noise).
        // Range: 0 .. 300 | Default: 0
        inline constexpr int  warmupFrames = 0;

        // Einmalige Kopfzeile.
        // Range: {false,true} | Default: true
        inline constexpr bool header       = true;

        // Auch die kompakte CUDA-[PERF]-Zeile emittieren?
        // Range: {false,true} | Default: false
        inline constexpr bool emitCudaLine = false;

        // Reserviert für noch kompakteres Format.
        // Range: {false,true} | Default: false
        inline constexpr bool compact      = false;
    }

// ============================== Framerate / VSync ============================
// Bildtakt (VSync bevorzugt, Cap als Fallback).
    // Main-Loop-Cap aktivieren.
    // Range: {false,true} | Default: true
    inline constexpr bool capFramerate = true;

    // Ziel-FPS bei aktivem Cap.
    // Range: 30 .. 240 | Default: 60
    inline constexpr int  capTargetFps = 60;

    // GL-VSync anfragen (Treiberhoheit möglich).
    // Range: {false,true} | Default: true
    inline constexpr bool preferVSync  = true;

// ============================== Interop / Upload =============================
// PBO-Ring (GL-Upload). Größer = weniger Stalls, mehr VRAM.
    // Anzahl PBOs im Ring.
    // Range: 3 .. 12 | Default: 8
    inline constexpr int pboRingSize = 8;

// ============================== Overlays / HUD ===============================
// Diagnostische Overlays.
    // Heatmap (Entropy/Contrast).
    // Range: {false,true} | Default: true
    inline constexpr bool  heatmapOverlayEnabled       = true;

    // Warzenschwein-HUD-Text (Stats/Status).
    // Range: {false,true} | Default: true
    inline constexpr bool  warzenschweinOverlayEnabled = true;

    // Textgröße (NDC-Skalierung).
    // Range: 0.0015 .. 0.004 | Default: 0.0025
    inline constexpr float hudPixelSize                = 0.0025f;

// ============================== Start / Window ===============================
    inline constexpr int   width      = 1024;  // Startauflösung X (px)
    inline constexpr int   height     = 768;   // Startauflösung Y (px)
    inline constexpr int   windowPosX = 100;   // Startposition X (px)
    inline constexpr int   windowPosY = 100;   // Startposition Y (px)
    inline constexpr float initialZoom    = 1.5f; // Startzoom
    inline constexpr float initialOffsetX = 0.0f; // Startoffset X (Complex)
    inline constexpr float initialOffsetY = 0.0f; // Startoffset Y (Complex)

// ============================== Iterations / Tiles ===========================
// Iterationsbudget & Compute-Tiles.
    // Startbudget für Iterationen.
    // Range: 50 .. 400 | Default: 100 | ↑ mehr Detail, ↓ FPS
    inline constexpr int INITIAL_ITERATIONS = 100;

    // Harte Obergrenze (Sicherheitsnetz).
    // Range: 10000 .. 200000 | Default: 50000
    inline constexpr int MAX_ITERATIONS_CAP = 50000;

    // Compute-Tile-Größe und Klammern (Kernels).
    // Constraint: MIN ≤ BASE ≤ MAX | Typisch 8..64 (Occupancy/Zoom).
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Mandelbrot Kernel ============================
// Standard-Blockgeometrie (X Vielfaches von 32).
    inline constexpr int MANDEL_BLOCK_X = 32;
    inline constexpr int MANDEL_BLOCK_Y = 8;

// ============================== Progressive / State ==========================
// Persistente Per-Pixel-Zustände (Resume, spätere Passes).
    // Master-Switch für Progressive-Features.
    // Range: {false,true} | Default: true
    inline constexpr bool progressiveEnabled = true;

// ============================== Kolibri / Grid ===============================
// Analyse-Grid wird in Screen-Pixels gehalten (zoom-invariant).
namespace Kolibri {
    // Screen-konstantes Grid aktivieren.
    // Range: {false,true} | Default: true
    inline constexpr bool gridScreenConstant = true;

    // Ziel-Tilegröße in Pixeln (Overlay/Metriken).
    // Range: 20 .. 40 | Default: 28 | ↑ gröber, ↓ feiner
    inline constexpr int  desiredTilePx      = 28;
}

// ============================== Stats Cadence ================================
// Heatmap-Berechnung nur alle N Frames (Dazwischen Reuse).
namespace StatsCadence {
    // 1 = jede Frame; 3 = gut balanciert; größer = weniger Last.
    // Range: 1 .. 16 | Default: 3
    inline constexpr int heatmapEveryN = 3;
}

// ============================== Target Bias ==================================
// Center-Bias für Zielauswahl (soft).
namespace TargetBias {
    // Bias aktivieren.
    // Range: {false,true} | Default: true
    inline constexpr bool   enabled  = true;

    // Gauß-Breite in NDC.
    // Range: 0.3 .. 1.2 | Default: 0.65 | ↓ engerer Center-Zug
    inline constexpr double sigmaNdc = 0.65;

    // Mischung raw vs. biased.
    // Range: 0 .. 1 | Default: 0.35 | ↑ mehr Bias-Gewicht
    inline constexpr double mix      = 0.35;
}

// ============================== Nav Bias (WASD/Arrows) =======================
// Sanfter Tastatur-Bias (additiv, kein Override).
namespace NavBias {
    // Master-Switch.
    inline constexpr bool   enabled     = true;

    // Aufbau-Tempo |bias|/s (dt-invariant).
    inline constexpr double gainPerSec  = 1.6;

    // Halbwertzeit des Abklingens (s).
    inline constexpr double halfLifeSec = 0.8;

    // Radialer Cap |bias| in NDC.
    inline constexpr double maxNdc      = 0.28;

    // Y leicht dämpfen (HUD-Lesbarkeit).
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
// Orbit/Policy/Color – Phase „Replikatoren“ (selbstlernend, Shadow aktiv).
namespace Perturb {
    // Perturbation-Pfad (Ctrl+P toggelt zur Laufzeit intern).
    // Range: {false,true} | Default: true
    inline constexpr bool   enabled       = true;

    // Schwellwert in Pixeln: unterhalb → Perturb on.
    // Range: 4 .. 24 | Default: 12
    inline constexpr int    gatePixelSize = 12;

    // Skalierung der Δ-Fehlerakkumulation.
    // Range: 0.25 .. 2.0 | Default: 1.0
    inline constexpr double deltaScale    = 1.0;

    // Sandbox-Tile für gezielte Tests (-1 = aus).
    // Range: {-1 oder gültiger Tile-Index} | Default: -1
    inline constexpr int    sandboxTile   = -1;
}

namespace Ai {
    // Master-Switch für AI-Pfade (AOP, Bandit-Telemetry).
    // Range: {false,true} | Default: true
    inline constexpr bool        enabled    = true;

    // Nur AOP-Controller (Policy-Telemetry) aktiv?
    // Range: {false,true} | Default: true
    inline constexpr bool        aopEnabled = true;

    // Ziel-Backend (nur Bezeichner; aktuell rein intern genutzt).
    // Range: {"cuda","dml","cpu"} | Default: "cuda"
    inline constexpr const char* ep         = "cuda";

    // Gewichtung E/C für einfache Scores (Overlay, Fallback).
    // Range: 0 .. 1 | Default: wE=0.60, wC=0.40
    inline constexpr float wE = 0.60f;
    inline constexpr float wC = 0.40f;

    // --- NEU: Soft-Coupling Controls (AI-Hint → sanft in Interest/NDC) ------
    // Master-Gate für sanftes Blending der AI-Empfehlung in die Zoom-Interest.
    inline constexpr bool  coupleEnabled = true;
    // Stärke des Blends bei „voller“ AI-Information (konstanter Faktor 0..1).
    inline constexpr float hintBlend     = 0.25f;
    // Mindestvertrauen (0..1), derzeit nur vorwärtskompatibel (ohne Effekt, wenn
    // keine Confidence-Telemetrie vorliegt).
    inline constexpr float minConfidence = 0.15f;
}

// *** Selbstlernender Bandit (LinUCB/RLS), on-device **************************
// Stages: 0=Shadow (lernt, steuert nicht), 1=Assisted (Kandidaten), 2=Auto (steuert Ziel).
namespace AiBandit {
    // Betriebsmodus.
    // Range: {0,1,2} | Default: 0 (Shadow sicher)
    inline constexpr int   stage            = 0;

    // Anzahl Kandidaten je Takt (Top-k).
    // Range: 1 .. 8 | Default: 3
    inline constexpr int   topK             = 3;

    // Frames zwischen Bandit-Updates.
    // Range: 1 .. 60 | Default: 5
    inline constexpr int   retargetInterval = 5;

    // UCB-Explorationsterm α.
    // Range: 0.1 .. 2.0 | Default: 0.8 | ↑ mehr Exploration
    inline constexpr float alpha            = 0.8f;

    // ε-Greedy-Anteil.
    // Range: 0 .. 0.3 | Default: 0.05 | ↑ mehr Zufall
    inline constexpr float epsilon          = 0.05f;

    // RLS-Regularisierung λ.
    // Range: 1e-6 .. 1e-1 | Default: 1e-2 | ↑ stabiler, ↓ träger
    inline constexpr float lambda           = 1.0e-2f;

    // Reward-Mischung E + β·C.
    // Range: 0 .. 2 | Default: 0.6 | ↑ mehr Gewicht auf C
    inline constexpr float beta             = 0.6f;

    // Reward-Clamp (robust gegen Ausreißer).
    // Range: [-5 .. 0], [0 .. 5] | Default: [-1,+1]
    inline constexpr float rewardClampLo    = -1.0f;
    inline constexpr float rewardClampHi    = +1.0f;

    // Deterministische RNG-Quelle.
    inline constexpr unsigned int seed      = 0xC0FFEEu;

    // Optionaler Persistenzpfad (derzeit nicht genutzt).
    inline constexpr const char* persistPath = "dist/ai/otter_bandit.bin";
}

// ============================== Luchs (Kompatibilität) ======================
// Kompat-Namespace für existierende Aufrufer (z. B. main.cpp, NVRTC-Pfad).
// nvrtc spiegelt das CMake-Flag OTTER_USE_NVRTC.
namespace Luchs {
    // Luchs-Subsystem sichtbar (Phase-1).
    inline constexpr bool enabled = true;

#if defined(OTTER_USE_NVRTC)
    // NVRTC zur Laufzeit verfügbar/aktiv?
    inline constexpr bool nvrtc   = true;
#else
    inline constexpr bool nvrtc   = false;
#endif

    // Kompilier-Budget (ms) für NVRTC-Shader (falls aktiv).
    // Range: 50 .. 1000 | Default: 200
    inline constexpr int  compileTimeout = 200;
}

} // namespace Settings
