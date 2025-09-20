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

    // ------------------------------------------------------------------------
    // Telemetrie-Dokumentation (Feldreihenfolge, stabil!)
    // ------------------------------------------------------------------------
    // [PERF] t,f,r,zm,it,fps,ai,cap,mx,ma,fr,df,map,md,en,ct,tx,ov,tt,e0,c0,tile,tiles,ring,skip,pbo,tex
    //  t=epoch-ms, f=frame, r=Res (WxH), zm=Zoom, it=MaxIter, fps=FPS,
    //  ai=AddIter (progressiveAddIter), cap=IterCap (maxIterations),
    //  mx=MaxFPS, ma/fre/df=mallocs/frees/devFlush,
    //  map=PBO map/unmap, md=mandelbrot kernel total, en/ct=entropy/contrast,
    //  tx=upload, ov=overlays, tt=frame total,
    //  e0/c0=last entropy/contrast, tile=tilePx, tiles=XxY,
    //  ring=PBO ring index, skip=no-upload, pbo/tex=GL ids.
    //
    // [PERT] active,refX,refY,iterLen,segSize,segCnt,deltaMax,rebases,store
    //  active (0/1), refX/refY=Referenzzentrum, iterLen=Orbit-Länge,
    //  segSize/segCnt=Segmentierung, deltaMax=beobachtetes |δ|,
    //  rebases=Rebase-Zähler, store=CONST|GLOBAL.

// ============================== Framerate / VSync ============================

    inline constexpr bool capFramerate = true; // CPU pacing (sleep+spin)
    inline constexpr int  capTargetFps = 60;
    inline constexpr bool preferVSync  = true;

// ============================== Interop / Upload =============================

    // HINWEIS: Muss RendererState::kPboRingSize entsprechen.
    inline constexpr int pboRingSize = 8;

// ============================== Overlays / HUD ===============================

    inline constexpr bool  heatmapOverlayEnabled       = true;
    inline constexpr bool  warzenschweinOverlayEnabled = true;
    inline constexpr float hudPixelSize                = 0.0025f;

// ============================== Start / Fenster ==============================

    inline constexpr int   width       = 1024;
    inline constexpr int   height      = 768;
    inline constexpr int   windowPosX  = 100;
    inline constexpr int   windowPosY  = 100;

    inline constexpr float initialZoom    = 1.5f;
    inline constexpr float initialOffsetX = -0.5f;
    inline constexpr float initialOffsetY = 0.0f;

// ============================== Iterationen / Tiles ==========================

    inline constexpr int INITIAL_ITERATIONS = 100;     // Startbudget
    inline constexpr int MAX_ITERATIONS_CAP = 50000;   // Safety clamp

    inline constexpr int BASE_TILE_SIZE = 32;
    inline constexpr int MIN_TILE_SIZE  = 8;
    inline constexpr int MAX_TILE_SIZE  = 64;

// ============================== Checks / Periodizität ========================

    inline constexpr bool   periodicityEnabled       = true;
    inline constexpr int    periodicityCheckInterval = 96;
    inline constexpr double periodicityEps2          = 1e-14;

// ============================== Progressive / State ==========================

    // Globaler Schalter für State-Puffer (Z, it). Kein Off-Switch im Renderpfad.
    inline constexpr bool progressiveEnabled = true;

    // Pro-Frame Iterationsbudget für Resume-Pfad (→ [PERF].ai)
    // Interaktiv: 16..48; Standard: 32..64; Offline: 96..256
    inline constexpr int  progressiveAddIter = 32;

// ============================== Mandelbrot Kernel ============================

    inline constexpr int  MANDEL_BLOCK_X = 32;
    inline constexpr int  MANDEL_BLOCK_Y = 8;
    inline constexpr int  MANDEL_UNROLL  = 4;
    inline constexpr bool MANDEL_USE_FMA = true;

// ============================== Kolibri/Grid =================================

namespace Kolibri {
    inline constexpr bool  gridScreenConstant  = true;
    inline constexpr int   desiredTilePx       = 28;
    inline constexpr bool  gridFadeEnable      = true;
    inline constexpr int   gridFadeMinFps      = 35;
    inline constexpr float gridFadeZoomStart   = 5000.0f;
} // namespace Kolibri

// ============================== Kolibri/Boost ================================
// Deep-zoom framerate stabilizer: frame-budget + runtime addIter bounds.
namespace KolibriBoost {
    inline constexpr bool   enable        = true;
    inline constexpr double targetFrameMs = 22.0;  // ≈45 FPS
    inline constexpr int    addIterMin    = 16;
    inline constexpr int    addIterMax    = 48;
    inline constexpr int    addIterStep   = 2;
} // namespace KolibriBoost

// ============================== Perturbation (Orbit) =========================
// Ein Pfad, sofortige Nutzung – gemäß "Genfer Großente v2.0".
// *Single source of truth*: Alle Pert-Schalter hier; keine Duplikate anderswo.
    // ------------------------------------------------------------------------
    // pertEnable
    // Wirkung: Aktiviert den Perturbation-Orbit-Pfad (Host-Upload + Kernel).
    // Empfehlung: false .. true (bool)
    // Effekt: true = Pert aktiviert; false = klassischer Mandelbrot-Pfad.
    // Hinweis: Codepfade müssen trotzdem sauber kompilieren.
    // ------------------------------------------------------------------------
    inline constexpr bool   pertEnable = true;

    // ------------------------------------------------------------------------
    // pertZoomMin
    // Wirkung: Unterhalb dieses Zooms bleibt Pert aus (klassischer Pfad).
    // Empfehlung: 1e4 .. 1e7 (double)
    // Effekt: Höher = späterer Pert-Einsatz; niedriger = früherer Einsatz.
    // ------------------------------------------------------------------------
    inline constexpr double pertZoomMin = 1e5;

    // ------------------------------------------------------------------------
    // zrefMaxLen
    // Wirkung: Maximalgröße des Referenz-Orbits (Anzahl double2 Elemente).
    // Empfehlung: ≤ 4096 für __constant__ (64 KiB), größere Werte erfordern
    //             GLOBAL-Store (Device-Mem) – Const-Array bleibt trotzdem
    //             auf zrefMaxLen dimensioniert.
    // Effekt: Höher = längere Referenzbahnen möglich; mehr Speicher.
    // ------------------------------------------------------------------------
    inline constexpr int    zrefMaxLen   = 4000;

    // ------------------------------------------------------------------------
    // zrefSegSize
    // Wirkung: Segmentgröße beim Orbit-Aufbau/Upload (Host), auch Telemetrie.
    // Empfehlung: 1024 .. 4096 (int), Potenzen von 2 bevorzugt.
    // Effekt: Größer = weniger Upload-Aufrufe, mehr Latenz je Segment.
    // ------------------------------------------------------------------------
    inline constexpr int    zrefSegSize  = 2048;

    // ------------------------------------------------------------------------
    // deltaMaxRebase
    // Wirkung: Rebase-Grenze für |δ| in *Pixelmaß* (pxScale). Bei
    //          |δ| > deltaMaxRebase * pxScale → Rebase erforderlich.
    // Empfehlung: 0.75 .. 2.0 (double)
    // Effekt: Kleiner = häufigere Rebases (stabiler, mehr Overhead);
    //         Größer  = seltener, Risiko von Pert-Fehlern steigt.
    // ------------------------------------------------------------------------
    inline constexpr double deltaMaxRebase = 1.25;

    // ------------------------------------------------------------------------
    // storeSwitchZoom  (Const → Global)
    // Wirkung: Ab diesem Zoom wird der Orbit grundsätzlich im GLOBAL-Speicher
    //          geführt (auch wenn zrefLen ≤ zrefMaxLen). Darunter bevorzugt
    //          CONST (sofern zrefLen ≤ zrefMaxLen), sonst GLOBAL.
    // Empfehlung: 2e5 .. 1e7 (double)
    // Effekt: Früher GLOBAL = konstante Pfadlatenz bei sehr tiefem Zoom.
    // ------------------------------------------------------------------------
    inline constexpr double storeSwitchZoom = 3e5;

    // ------------------------------------------------------------------------
    // deltaGuardAbs
    // Wirkung: Abbruchschwelle |δ| für Perturbations-Telemetrie/Fallback
    //          (in „Fraktal-Einheiten“). Sobald |δ| diese Schwelle überschreitet,
    //          darf der Gerätepytorch laute Fallbacks triggern (in diesem Schritt:
    //          nur Telemetrie, keine Bildänderung).
    // Empfehlung: 2.0 .. 16.0 (double)
    // Default: 4.0
    // ------------------------------------------------------------------------
    inline constexpr double deltaGuardAbs = 4.0;

    // ------------------------------------------------------------------------
    // pertDevLogEvery
    // Wirkung: Rate-Limit für optionale [PERT][DEV]-Blocklogs im Kernel.
    // Empfehlung: 30 .. 240 (int)
    // Default: 60
    // ------------------------------------------------------------------------
    inline constexpr int    pertDevLogEvery = 60;

    // Hinweis: Ehemalige Einstellungen "perturbSegSize", "perturbUseConstUpTo",
    // "perturbDeltaGuardK", "perturbRebaseDeltaPx" sind ersetzt/vereinheitlicht
    // durch zrefSegSize, zrefMaxLen, storeSwitchZoom und deltaMaxRebase.

} // namespace Settings
