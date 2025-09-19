///// Otter: Central config; every value documented (purpose, range, default).
///// Schneefuchs: No hidden macros; single source of truth for flags.
///// Maus: performanceLogging=1, ForceAlwaysZoom=1 baseline; ASCII-only logs.
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

// ============================== Interop / Upload =============================

    // ------------------------------------------------------------------------
    // pboRingSize
    // Wirkung: Anzahl der Pixel Buffer Objects (PBO) im Upload-Ring
    //          für CUDA <-> OpenGL Streaming.
    // Empfehlung:
    //   - typisch: 3 .. 12
    //   - Sweet-Spot: 4 .. 8 (Stall-Resistenz vs. VRAM-Verbrauch)
    //   - 16+ nur bei dokumentierten Treiber-Spikes und genügend VRAM
    // Effekt:
    //   - größer  = seltener Map-Stalls, aber mehr VRAM-Bedarf
    //   - kleiner = weniger VRAM, höhere Wahrscheinlichkeit für Busy-PBO
    // Speicherbedarf pro Frame:
    //   bytes ≈ width * height * bytesPerPixel * pboRingSize
    // Default hier: 8 (balanciert für 1080p/1440p).
    // ------------------------------------------------------------------------
    inline constexpr int pboRingSize = 8;

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
    // Empfehlung: false .. true (bool) – standardmäßig aus, erst testen/tunen.
    // Effekt: true = weniger Iterationen bei bounded Orbits; false = unverändert.
    // ------------------------------------------------------------------------
    inline constexpr bool periodicityEnabled = true;

    // ------------------------------------------------------------------------
    // progressiveAddIter
    // Wirkung: Progressives Iterations-Budget pro Frame und Pixel (nur im Resume-Pfad).
    // Bedeutung:
    //   - Dieser Wert bestimmt die verpflichtende Rechenarbeit jedes Frames:
    //       cost_pro_frame  ~  width * height * progressiveAddIter
    //     (zzgl. kleiner Overheads). Bei 1024x768 und 32 ⇒ ~25.2 Mio. Iterationen/Frame.
    //   - Hoeherer Wert = schneller konvergierende Kanten/Alpha, aber niedrigere FPS.
    //   - Niedrigerer Wert = hoehere FPS, dafuer sichtbarere "Weichheit", die ueber mehrere
    //     Frames verschwindet.
    // Praxisrichtwerte:
    //   - Interaktiv (GPU geteilt / Hintergrundlast):   16 .. 48
    //   - Standard-Sichtprobe (1024x768, Solo-GPU):     32 .. 64
    //   - Offline / "fix Bild in wenigen Frames":       96 .. 256
    // Hinweise:
    //   - Wirkt nur, wenn progressiveEnabled==true und die State-Puffer aktiv sind.
    //   - MAX_ITERATIONS_CAP bleibt die absolute Obergrenze pro Pixel.
    //   - Debug-/Perf-Logging kann den wahrgenommenen Konvergenztakt beeinflussen,
    //     die reine Iterationsmenge bleibt jedoch durch diesen Wert definiert.
    // ------------------------------------------------------------------------
    inline constexpr int  progressiveAddIter = 32;

    // ------------------------------------------------------------------------
    // periodicityCheckInterval
    // Wirkung: Prüfintervall N (Iterationen) zwischen zwei Proben von z.
    // Empfehlung: 32 .. 128 (int) – größer = seltener, schneller; kleiner = häufiger, genauer.
    // Hinweis: Kolibri/Boost empfiehlt moderat höhere Intervalle. Default hier: 96.
    // ------------------------------------------------------------------------
    inline constexpr int periodicityCheckInterval = 96;

    // ------------------------------------------------------------------------
    // periodicityEps2
    // Wirkung: Schwellwert für Abstand² zwischen z-Proben (kleiner = strenger).
    // Empfehlung: 1e-16 .. 1e-12 (double) – Startwert konservativ.
    // ------------------------------------------------------------------------
    inline constexpr double periodicityEps2 = 1e-14;

// ============================== Progressive / State ==========================

    // ------------------------------------------------------------------------
    // progressiveEnabled
    // Wirkung: Allokiert persistente Per-Pixel-States (Z, it) für progressive Iteration.
    // Hinweis: In Keks 4 werden die Puffer nur angelegt (Renderpfad folgt in Keks 6/7).
    // **AN**: true
    // ------------------------------------------------------------------------
    inline constexpr bool progressiveEnabled = true;


// ============================== Mandelbrot Kernel ============================
// Block geometry (affects occupancy & coalescing).
// Recommendation: (32,8) balanced; (32,16) higher ILP (watch registers).
// ----------------------------------------------------------------------------
inline constexpr int MANDEL_BLOCK_X = 32;   // threads in X (multiple of 32)
inline constexpr int MANDEL_BLOCK_Y = 8;    // threads in Y (tune vs. registers)

// Unroll hint for inner iteration loop (compute-bound).
// 1..8; 4 is a safe sweet-spot.
inline constexpr int MANDEL_UNROLL  = 4;

// Enable fused multiply-add in the iteration updates (zy and zx path).
inline constexpr bool MANDEL_USE_FMA = true;


// ============================== Kolibri/Grid =================================
// Screen-constant heatmap tiles (visual analysis grid independent of zoom).
// Implementierung: frame_pipeline berechnet tileSizePx aus Fenstergröße.
// -----------------------------------------------------------------------------
namespace Kolibri {
    // ------------------------------------------------------------------------
    // gridScreenConstant
    // Wirkung: Aktiviert screen-konstante Tiles im Analyse-/Overlay-Pfad.
    // Empfehlung: true (bool)
    // Effekt: Kacheln bleiben ~gleich groß in Pixeln, unabhängig vom Zoom.
    // ------------------------------------------------------------------------
    inline constexpr bool gridScreenConstant = true;

    // ------------------------------------------------------------------------
    // desiredTilePx
    // Wirkung: Zielgröße einer Kachel in Bildschirm-Pixeln.
    // Empfehlung: 20 .. 40 (int) – 28 ist ein guter Startwert.
    // Effekt: kleiner = dichteres Raster, größer = weniger Kacheln.
    // ------------------------------------------------------------------------
    inline constexpr int  desiredTilePx = 28;

    // ------------------------------------------------------------------------
    // gridFadeEnable / gridFadeMinFps / gridFadeZoomStart
    // Wirkung: Optionaler Fade des Gitters, wenn es „stören“ würde.
    // Empfehlung:
    //   - gridFadeEnable: true
    //   - gridFadeMinFps: 35 .. 45
    //   - gridFadeZoomStart: szenabhaengig, z.B. 5000.0f
    // Effekt: blendet Raster sanft zurück, wenn FPS niedrig sind oder Zoom hoch ist.
    // ------------------------------------------------------------------------
    inline constexpr bool  gridFadeEnable   = true;
    inline constexpr int   gridFadeMinFps   = 35;
    inline constexpr float gridFadeZoomStart = 5000.0f;
} // namespace Kolibri


// ============================== Kolibri/Boost ================================
// Deep-zoom framerate stabilizer: frame-budget + runtime addIter bounds.
// Pipeline verwendet dt->addIterRuntime in Resume-Pfad; Kernel-Tuning separat.
// -----------------------------------------------------------------------------
namespace KolibriBoost {
    // ------------------------------------------------------------------------
    // enable
    // Wirkung: Schaltet Budget-Regler frei (dt->addIterRuntime) und Kernel-Tuning.
    // Empfehlung: true (bool)
    // ------------------------------------------------------------------------
    inline constexpr bool   enable = true;

    // ------------------------------------------------------------------------
    // targetFrameMs
    // Wirkung: Zielzeit pro Frame (Budget). Pipeline senkt/erhöht addIterRuntime,
    // um diese Zeit grob einzuhalten.
    // Empfehlung: 18.0 .. 25.0 (double) – 22.0 ≈ 45 FPS gefühlt sehr flüssig.
    // ------------------------------------------------------------------------
    inline constexpr double targetFrameMs = 22.0;

    // ------------------------------------------------------------------------
    // addIterMin / addIterMax / addIterStep
    // Wirkung: Klemmen und Schrittweite für addIterRuntime (pro Frame).
    // Empfehlung: Min 12..24, Max 40..64, Step 1..4
    // Effekt: Tighter Min -> höhere FPS-Sicherheit; höheres Max -> schnellere Schärfe.
    // ------------------------------------------------------------------------
    inline constexpr int addIterMin  = 16;
    inline constexpr int addIterMax  = 48;
    inline constexpr int addIterStep = 2;
} // namespace KolibriBoost

} // namespace Settings
