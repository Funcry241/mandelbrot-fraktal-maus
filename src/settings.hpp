///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/settings.hpp
#pragma once

// ============================================================================
// Central project settings – fully documented (nur aktive/benutzte Schalter).
// Policy: All runtime LOG/DEBUG output must be English and ASCII-only.
// Keine versteckten Semantikänderungen. Werte stabil.
// ============================================================================

namespace Settings {

// ============================== Zoom / Planner ===============================

    // ------------------------------------------------------------------------
    // ForceAlwaysZoom
    // Wirkung: Erzwingt kontinuierliches Zoomen, unabhängig von Entropie/Kontrast.
    // Empfehlung: false .. true  (bool)
    // Effekt: true = stetige Bewegung (Demo/Drift-Fallback); false = rein analysegetrieben.
    // ------------------------------------------------------------------------
    inline constexpr bool   ForceAlwaysZoom = true;

    // ------------------------------------------------------------------------
    // warmUpFreezeSeconds
    // Wirkung: Nach Retarget für diese Zeit keine Richtungswechsel (Stabilisierung).
    // Empfehlung: 0.2 .. 2.0 (double)
    // Effekt: höher = ruhiger, träger; niedriger = agiler, evtl. Flattern.
    // ------------------------------------------------------------------------
    inline constexpr double warmUpFreezeSeconds = 1.0;

// ============================== Logging / Perf ===============================

    // ------------------------------------------------------------------------
    // debugLogging
    // Wirkung: Aktiviert gezielte Debug-/Diagnose-Ausgaben (Host/Device).
    // Empfehlung: false .. true (bool)
    // Effekt: true = mehr Einblick, leicht geringere FPS.
    // ------------------------------------------------------------------------
    inline constexpr bool debugLogging  = true;

    // ------------------------------------------------------------------------
    // performanceLogging
    // Wirkung: Verdichtete [PERF]-Logs entlang der Frame-Pipeline.
    // Empfehlung: false .. true (bool)
    // Effekt: true = periodische Metriken; false = still.
    // ------------------------------------------------------------------------
    inline constexpr bool performanceLogging = true;

// ============================== Framerate / VSync ============================

    // ------------------------------------------------------------------------
    // capFramerate
    // Wirkung: CPU-seitige Framerate-Begrenzung (sleep+spin) auf Ziel-FPS.
    // Empfehlung: false .. true (bool)
    // Effekt: true = stabileres Pacing, weniger Jitter; false = volles Tempo.
    // ------------------------------------------------------------------------
    inline constexpr bool capFramerate = true;

    // ------------------------------------------------------------------------
    // capTargetFps
    // Wirkung: Zielrate der Framerate-Begrenzung.
    // Empfehlung: 30 .. 240 (int)
    // Effekt: höher = höheres Zieltempo (CPU-lastiger), niedriger = ruhiger.
    // ------------------------------------------------------------------------
    inline constexpr int  capTargetFps = 60;

    // ------------------------------------------------------------------------
    // preferVSync
    // Wirkung: Wenn verfügbar, VSync bevorzugen (GPU/Swap-gekoppelt).
    // Empfehlung: false .. true (bool)
    // Effekt: true = gleichmäßiges Bild, evtl. mehr Latenz; false = unabhängig.
    // ------------------------------------------------------------------------
    inline constexpr bool preferVSync = true;

// ============================== Overlays / HUD ===============================

    // ------------------------------------------------------------------------
    // heatmapOverlayEnabled
    // Wirkung: Heatmap-Overlay einschalten.
    // Empfehlung: false .. true (bool)
    // ------------------------------------------------------------------------
    inline constexpr bool heatmapOverlayEnabled = true;

    // ------------------------------------------------------------------------
    // warzenschweinOverlayEnabled
    // Wirkung: Warzenschwein-HUD einschalten.
    // Empfehlung: false .. true (bool)
    // ------------------------------------------------------------------------
    inline constexpr bool warzenschweinOverlayEnabled = true;

    // ------------------------------------------------------------------------
    // hudPixelSize
    // Wirkung: Grundgröße der HUD-Pixel (NDC-basiert).
    // Empfehlung: 0.0015f .. 0.0040f (float)
    // Effekt: größer = besser lesbar, aber präsenter.
    // ------------------------------------------------------------------------
    inline constexpr float hudPixelSize = 0.0025f;

// ============================== Start / Fenster ==============================

    // ------------------------------------------------------------------------
    // width / height
    // Wirkung: Startauflösung des Fensters (Pixel).
    // Empfehlung: 800x600 .. 3840x2160
    // ------------------------------------------------------------------------
    inline constexpr int width       = 1024;
    inline constexpr int height      = 768;

    // ------------------------------------------------------------------------
    // windowPosX / windowPosY
    // Wirkung: Startposition des Fensters.
    // Empfehlung: >= 0 (int)
    // ------------------------------------------------------------------------
    inline constexpr int windowPosX  = 100;
    inline constexpr int windowPosY  = 100;

    // ------------------------------------------------------------------------
    // initialZoom / initialOffsetX / initialOffsetY
    // Wirkung: Startwerte der Ansicht (Zoom + Verschiebung).
    // Empfehlung: projekt-/motivabhängig
    // ------------------------------------------------------------------------
    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = -0.5f;
    inline constexpr float initialOffsetY = 0.0f;

// ============================== Iterationen / Tiles ==========================

    // ------------------------------------------------------------------------
    // INITIAL_ITERATIONS
    // Wirkung: Startbudget Iterationen pro Pixel (wird dynamisch erhöht).
    // Empfehlung: 50 .. 400 (int)
    // ------------------------------------------------------------------------
    inline constexpr int INITIAL_ITERATIONS = 100;

    // ------------------------------------------------------------------------
    // MAX_ITERATIONS_CAP
    // Wirkung: Harte Obergrenze für Iterationen/Pixel (Safety/Clamp).
    // Empfehlung: 10'000 .. 200'000 (int) – abhängig von GPU/Zoomtiefe.
    // ------------------------------------------------------------------------
    inline constexpr int MAX_ITERATIONS_CAP = 50000;

    // ------------------------------------------------------------------------
    // BASE/MIN/MAX_TILE_SIZE
    // Wirkung: Kachelgrößen für CUDA-Kernels (Arbeitsaufteilung).
    // Empfehlung: MIN ≤ BASE ≤ MAX; typ. 8/32/64.
    // ------------------------------------------------------------------------
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Checks / Periodizität ========================

    // ------------------------------------------------------------------------
    // periodicityEnabled
    // Wirkung: Aktiviert Periodizitäts-Probe im Kernel (frühzeitiger Abbruch bei (nahe) zyklischer Bahn).
    // Effekt: true = weniger Iterationen bei bounded Orbits; false = unverändert.
    // ------------------------------------------------------------------------
    inline constexpr bool   periodicityEnabled     = true;

    // ------------------------------------------------------------------------
    // periodicityCheckInterval
    // Wirkung: Prüfintervall N (Iterationen) zwischen zwei Proben von z.
    // Empfehlung: 32 .. 128 (int)
    // ------------------------------------------------------------------------
    inline constexpr int    periodicityCheckInterval = 64;

    // ------------------------------------------------------------------------
    // periodicityEps2
    // Wirkung: Schwellwert für Abstand² zwischen z-Proben (kleiner = strenger).
    // Empfehlung: 1e-16 .. 1e-12 (double)
    // ------------------------------------------------------------------------
    inline constexpr double periodicityEps2        = 1e-14;

// ============================== Progressive / State ==========================

    // ------------------------------------------------------------------------
    // progressiveEnabled
    // Wirkung: Allokiert persistente Per-Pixel-States (Z, it) für progressive Iteration.
    // **AN**: true
    // ------------------------------------------------------------------------
    inline constexpr bool progressiveEnabled = true;

    // ------------------------------------------------------------------------
    // progressiveAddIter
    // Wirkung: Inkrementelles Iterationsbudget pro Frame (nur bei progressiveEnabled).
    // Empfehlung: 64 .. 256 (int)
    // ------------------------------------------------------------------------
    inline constexpr int  progressiveAddIter = 128;

} // namespace Settings
