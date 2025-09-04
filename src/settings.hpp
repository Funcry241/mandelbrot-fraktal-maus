///// Otter: settings.hpp - voll dokumentierte Toggles; ASCII-only; deterministisch.
///// Schneefuchs: Keine versteckten Semantik-Aenderungen; Defaults stabil; /WX-fest.
///// Maus: Kein Logging hier; nur Konstanten & Doku; Header-only; Nacktmull-Pullover aktiv.
///// Datei: src/settings.hpp

#pragma once

// ============================================================================
// Central project settings - fully documented.
// Policy: All runtime LOG/DEBUG output must be English and ASCII-only.
// Kommentare duerfen Deutsch sein. Referenzen:
//   - Otter = vom User inspiriert (Bezug zu Otter)
//   - Schneefuchs = von Schwester inspiriert (Bezug zu Schneefuchs)
// Keine Regressionen, keine versteckten Semantik-Aenderungen. Werte stabil.
// Neu (Otter/Schneefuchs): Framerate-Cap (60 FPS), optionales VSync,
// sowie Nacktmull-Pullover (Praezisions- und Orbit-Guards).
// ============================================================================

namespace Settings {

    // ------------------------------------------------------------------------
    // ForceAlwaysZoom
    // Wirkung:
    //   Erzwingt kontinuierliches Zoomen, unabhaengig von Entropie/Kontrast.
    //
    // Empfehlung (Min..Max):
    //   false .. true  (bool)
    //
    // Effekt bei Erhoehung/Verringerung:
    //   - true: Stetige Bewegung, ideal fuer Demos und als Drift-Fallback. (Otter)
    //   - false: Bewegung nur aus Analysewerten; ehrlicher, aber stoppanfaelliger.
    // ------------------------------------------------------------------------
    inline constexpr bool   ForceAlwaysZoom = true;   // Otter: always zoom

    // ------------------------------------------------------------------------
    // ForcedZoomStep
    // Wirkung:
    //   Multiplikativer Zoomfaktor pro Frame, falls ForceAlwaysZoom aktiv ist.
    //   < 1.0 zoomt rein, > 1.0 zoomt raus.
    //
    // Empfehlung (Min..Max):
    //   0.90 .. 0.999 (double)
    //
    // Effekt:
    //   - Naeher an 1.0: langsamer, subtiler Zoom.
    //   - Richtung 0.90: schneller Zoom, potenziell unruhiger. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr double ForcedZoomStep  = 0.97;

    // ------------------------------------------------------------------------
    // debugLogging
    // Wirkung:
    //   Aktiviert gezielte Debug-/Diagnose-Ausgaben (Host/Device).
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Mehr Einblick, leicht geringere FPS.
    //   - false: Ruhiger Lauf, ideal fuer Captures/Benchmarks. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool debugLogging  = true;

    // ------------------------------------------------------------------------
    // performanceLogging
    // Wirkung:
    //   Aktiviert kompakte Performance-Logs (Zeiten pro Phase, FPS, Metriken).
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Praezise Einsicht; Overhead ca. 1-2 %.
    //   - false: Maximale Ruhe. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool performanceLogging = true;

    // ------------------------------------------------------------------------
    // progressiveEnabled  (NEU)
    // Wirkung:
    //   Aktiviert Progressive Iteration + Resume (zeitbudgetierte Slices).
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Progressive-Pfad nutzbar.
    //   - false: Klassischer Single-Pass. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool progressiveEnabled = true;

    // ------------------------------------------------------------------------
    // capFramerate  (NEU)
    // Wirkung:
    //   CPU-Seitenratenbegrenzung per sleep+spin auf Ziel-FPS.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Stabileres Pacing, weniger Jitter. (Otter)
    //   - false: Volles Tempo fuer Benchmarks.
    // ------------------------------------------------------------------------
    inline constexpr bool capFramerate = true;

    // ------------------------------------------------------------------------
    // capTargetFps  (NEU)
    // Wirkung:
    //   Zielrate der Framebegrenzung.
    //
    // Empfehlung (Min..Max):
    //   30 .. 240
    //
    // Effekt:
    //   - Hoeher: straffere Zielrate, mehr Last.
    //   - Niedriger: mehr Ruhezeit, weniger Last. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr int capTargetFps = 60;

    // ------------------------------------------------------------------------
    // preferVSync  (NEU)
    // Wirkung:
    //   Setzt glfwSwapInterval(1) fuer tear-freies Rendering.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Tear-frei, natuerliche Praesentation; etwas mehr Latenz moeglich.
    //   - false: Kein VSync, nur Software-Pacing. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool preferVSync = true;

    // ======================= Nacktmull-Pullover ==============================
    // Ziel:
    //   Praezisionssichere Mapping-Pipeline und orbit-sichere Retarget-Strategie.
    //   Verhindert Pixel-Quantisierung bei tiefem Zoom und stale Orbits.
    // ========================================================================

    // ------------------------------------------------------------------------
    // nacktmullPulloverEnabled  (NEU)
    // Wirkung:
    //   Schaltet alle Schutzmechanismen in diesem Block logisch zusammen.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Precision-Guard, Rebase, Orbit-Refresh nutzbar.
    //   - false: Schutzpfade aus; klassisches Verhalten.
    // ------------------------------------------------------------------------
    inline constexpr bool nacktmullPulloverEnabled = true;

    // ------------------------------------------------------------------------
    // useDoublePrecisionPipeline  (NEU)
    // Wirkung:
    //   Erzwingt double fuer Mapping/Center/PixelScale und Orbit-Parameter
    //   (Host- und Device-Args), wo unterstuetzt.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Deutlich hoehere numerische Reserve im Deep-Zoom.
    //   - false: Schneller, aber frueher Quantisierung. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool useDoublePrecisionPipeline = true;

    // ------------------------------------------------------------------------
    // precisionGuardEnabled  (NEU)
    // Wirkung:
    //   Aktiviert Nachbarpixel-Delta-vs-ULP-Pruefung im Hostpfad.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Logik kann Rebase/Orbit-Refresh triggern.
    //   - false: Keine Schutzaktion, groesstes Risiko fuer Pixel-Blocks.
    // ------------------------------------------------------------------------
    inline constexpr bool precisionGuardEnabled = true;

    // ------------------------------------------------------------------------
    // precisionGuardMinRatio  (NEU)
    // Wirkung:
    //   Schwelle fuer (delta_cx / ulp(cx)). Unterhalb dieser Ratio ist die
    //   Float-Todeszone erreicht und eine Gegenmassnahme angezeigt.
    //
    // Empfehlung (Min..Max):
    //   2.0 .. 16.0  (double)
    //
    // Effekt:
    //   - Hoeher: Frueheres Eingreifen (mehr Rebase/Refresh).
    //   - Niedriger: Spaeteres Eingreifen (weniger Schutzaktionen). (Otter)
    // ------------------------------------------------------------------------
    inline constexpr double precisionGuardMinRatio = 4.0;

    // ------------------------------------------------------------------------
    // rebaseEnable  (NEU)
    // Wirkung:
    //   Erlaubt periodisches Re-Centering, um Ausloeschung zu vermeiden.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Center wird in die Naehe von 0 verschoben, Praezision steigt.
    //   - false: Kein Rebase, hoeheres Risiko fuer Praezisionsverlust.
    // ------------------------------------------------------------------------
    inline constexpr bool rebaseEnable = true;

    // ------------------------------------------------------------------------
    // rebaseCenterOverPixelStepThreshold  (NEU)
    // Wirkung:
    //   Rebase-Trigger, wenn |center| / pixel_step diesen Wert ueberschreitet.
    //
    // Empfehlung (Min..Max):
    //   1e6 .. 1e9  (double)
    //
    // Effekt:
    //   - Hoeher: Seltener Rebase, mehr Praezisionsrisiko.
    //   - Niedriger: Haeufiger Rebase, stabilere Praezision. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr double rebaseCenterOverPixelStepThreshold = 1.0e7;

    // ------------------------------------------------------------------------
    // orbitPerTileEnabled  (NEU)
    // Wirkung:
    //   Nutzt tile-lokale Referenzorbits statt eines globalen pro Frame.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Robust in Randbereichen grosser Bilder, weniger Drift.
    //   - false: Einfacher Pfad, moeglichere Randartefakte. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr bool orbitPerTileEnabled = false;

    // ------------------------------------------------------------------------
    // orbitRebuildOnRetarget  (NEU)
    // Wirkung:
    //   Erzwingt Orbits-Neuaufbau nach Retarget plus Warm-up-Freeze.
    //
    // Empfehlung:
    //   false .. true (bool)
    //
    // Effekt:
    //   - true: Exakte Orbit-Basis nach Zielwechsel.
    //   - false: Orbit bleibt bestehen, Risiko fuer stale Basis. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool orbitRebuildOnRetarget = true;

    // ------------------------------------------------------------------------
    // orbitRebuildMaxDeltaOverPixelStep  (NEU)
    // Wirkung:
    //   Trigger fuer Orbit-Neuaufbau, wenn max(|delta_c|)/pixel_step diese
    //   Schwelle ueberschreitet.
    //
    // Empfehlung (Min..Max):
    //   1e4 .. 1e8  (double)
    //
    // Effekt:
    //   - Hoeher: Weniger Rebuilds, groesseres Drift-Risiko.
    //   - Niedriger: Mehr Rebuilds, stabilere Perturbation.
    // ------------------------------------------------------------------------
    inline constexpr double orbitRebuildMaxDeltaOverPixelStep = 1.0e6;

    // ------------------------------------------------------------------------
    // deltaGuardFactorBeta  (NEU)
    // Wirkung:
    //   Device-seitiger Guard: falls |delta_z| > beta * pixel_step, Flag fuer
    //   Rebase/Refresh setzen.
    //
    // Empfehlung (Min..Max):
    //   8 .. 256  (double)
    //
    // Effekt:
    //   - Hoeher: Weniger Flags, Risiko fuer Drift.
    //   - Niedriger: Mehr Flags, stabiler aber event. mehr Rebuilds. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr double deltaGuardFactorBeta = 64.0;

    // ======================= Zoom Silk-Lite / Planner ========================

    // ------------------------------------------------------------------------
    // warmUpFreezeSeconds
    // Wirkung:
    //   Nach Retarget fuer diese Zeit keine Richtungswechsel (Stabilisierung).
    //
    // Empfehlung (Min..Max):
    //   0.2 .. 2.0 (double)
    //
    // Effekt:
    //   - Hoeher: Ruhiger, traeger Start.
    //   - Niedriger: Schnelleres Reagieren, evtl. Flattern. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr double warmUpFreezeSeconds = 1.0;

    // ------------------------------------------------------------------------
    // retargetIntervalFrames
    // Wirkung:
    //   Mindestabstand zwischen neuen Zielauswahlen (Frames).
    //
    // Empfehlung (Min..Max):
    //   1 .. 30 (int)
    //
    // Effekt:
    //   - Hoeher: Weniger Zielwechsel, stabiler Fokus.
    //   - Niedriger: Reaktiver, aber potentiell nervoes. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr int retargetIntervalFrames = 5;

    // ------------------------------------------------------------------------
    // lockFrames
    // Wirkung:
    //   Harte Sperre der Zielwahl fuer N Frames nach Lock.
    //
    // Empfehlung (Min..Max):
    //   4 .. 60 (int)
    //
    // Effekt:
    //   - Hoeher: Konstanter Fokus, weniger Zittern.
    //   - Niedriger: Mehr Reaktivitaet. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr int lockFrames = 12;

    // ------------------------------------------------------------------------
    // hysteresisRel
    // Wirkung:
    //   Relative Ueberlegenheit, die ein Kandidat gegen den aktuellen Sieger
    //   aufweisen muss, um einen Wechsel auszueloesen (z. B. 0.12 = +12 %).
    //
    // Empfehlung (Min..Max):
    //   0.02 .. 0.30 (double)
    //
    // Effekt:
    //   - Hoeher: Weniger Wechsel, stabiler.
    //   - Niedriger: Schnelleres Umschalten. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr double hysteresisRel = 0.12;

    // ------------------------------------------------------------------------
    // softmaxTopK
    // Wirkung:
    //   Anzahl Top-K Tiles/Kandidaten fuer Softmax-basierte Zielwahl.
    //
    // Empfehlung (Min..Max):
    //   1 .. 32 (int)
    //
    // Effekt:
    //   - Hoeher: Breitere Auswahl, robust gegen Ausreisser.
    //   - Niedriger: Fokus auf Spitzenkandidaten, evtl. sprunghaft. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr int softmaxTopK = 6;

    // ------------------------------------------------------------------------
    // statsCadenceFrames
    // Wirkung:
    //   Frequenz fuer Statistik-Aktualisierung (Median/MAD, Normalisierung).
    //
    // Empfehlung (Min..Max):
    //   1 .. 10 (int)
    //
    // Effekt:
    //   - Hoeher: Weniger CPU-Overhead, traegere Reaktion.
    //   - Niedriger: Reaktiver, mehr Overhead. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr int statsCadenceFrames = 3;

    // ------------------------------------------------------------------------
    // pdKp, pdKd
    // Wirkung:
    //   Proportional- und Damping-Gewichte fuer den Zoom-Motion-Planner.
    //
    // Empfehlung (Min..Max):
    //   Kp: 0.1 .. 3.0
    //   Kd: 0.0 .. 1.5
    //
    // Effekt:
    //   - Kp hoeher: Schnelleres Anziehen, Risiko Ueberschwingen.
    //   - Kd hoeher: Mehr Daempfung, ruhiger, evtl. traeger. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr double pdKp = 1.0;
    inline constexpr double pdKd = 0.30;

    // ------------------------------------------------------------------------
    // pdMaxAccel, pdMaxVelocity
    // Wirkung:
    //   Klemmen der Planner-Ausgaben fuer sanftes Pacing.
    //
    // Empfehlung (Min..Max):
    //   Accel: 0.1 .. 10.0
    //   Vel  : 0.01 .. 5.0
    //
    // Effekt:
    //   - Hoeher: Reaktiver, evtl. ruckeliger.
    //   - Niedriger: Ruhiger, evtl. zaeh. (Otter)
    // ------------------------------------------------------------------------
    inline constexpr double pdMaxAccel   = 3.0;
    inline constexpr double pdMaxVelocity= 1.2;

    // ------------------------------------------------------------------------
    // microDeadbandNDC
    // Wirkung:
    //   Deadband in NDC fuer Mikro-Bewegungen, um Flattern zu vermeiden.
    //
    // Empfehlung (Min..Max):
    //   1e-5 .. 5e-3
    //
    // Effekt:
    //   - Hoeher: Weniger Zittern, aber geringere Feinausrichtung.
    //   - Niedriger: Praeziser, aber potentiell vibrationen. (Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr double microDeadbandNDC = 5e-4;

    // ======================= UI / Overlays / Window ==========================

    // ------------------------------------------------------------------------
    // heatmapOverlayEnabled
    // Wirkung:
    //   Sichtbarkeit des Heatmap-Overlays beim Start.
    // ------------------------------------------------------------------------
    inline constexpr bool heatmapOverlayEnabled = true;

    // ------------------------------------------------------------------------
    // warzenschweinOverlayEnabled
    // Wirkung:
    //   Sichtbarkeit des Warzenschwein-HUD beim Start.
    // ------------------------------------------------------------------------
    inline constexpr bool warzenschweinOverlayEnabled = true;

    // ------------------------------------------------------------------------
    // hudPixelSize
    // Wirkung:
    //   Skalierung der HUD-Glyphen in NDC-Einheiten.
    //
    // Empfehlung (Min..Max):
    //   0.0015f .. 0.0040f
    // ------------------------------------------------------------------------
    inline constexpr float hudPixelSize = 0.0025f;

    // ------------------------------------------------------------------------
    // Fensterkonfiguration
    // Wirkung:
    //   Startgroesse und -position des Fensters.
    // ------------------------------------------------------------------------
    inline constexpr int width       = 1024;
    inline constexpr int height      = 768;
    inline constexpr int windowPosX  = 100;
    inline constexpr int windowPosY  = 100;

    // ------------------------------------------------------------------------
    // Initialer Fraktal-Ausschnitt
    // Wirkung:
    //   Start-Zoom und Offset im Komplexraum.
    // ------------------------------------------------------------------------
    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = -0.5f;
    inline constexpr float initialOffsetY = 0.0f;

    // ------------------------------------------------------------------------
    // Iterationssteuerung
    // Wirkung:
    //   Startbudget und harte Obergrenze der Iterationen pro Pixel.
    // ------------------------------------------------------------------------
    inline constexpr int INITIAL_ITERATIONS = 100;
    inline constexpr int MAX_ITERATIONS_CAP = 50000;

    // ------------------------------------------------------------------------
    // CUDA Tile-Groessen
    // Wirkung:
    //   Arbeitsaufteilung fuer CUDA-Kernels.
    // ------------------------------------------------------------------------
    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

} // namespace Settings
