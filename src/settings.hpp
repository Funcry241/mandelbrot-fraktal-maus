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

    inline constexpr bool   periodicityEnabled      = true;
    inline constexpr int    periodicityCheckInterval= 96;
    inline constexpr double periodicityEps2         = 1e-14;

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

// ============================== Perturbation =================================
// Always-on Orbit-Perturbation Parameter (kein Off-Switch im Build).
// Host/Kernel verwenden EINEN kanonischen pxScale (siehe FramePipeline/coords).
    // Segmentgröße für den Referenz-Orbit (Host-Aufbau/Upload).
    inline constexpr int    perturbSegSize        = 2048;   // 1024..4096

    // Schwelle (Elemente) bis zu der __constant__ Speicher bevorzugt wird,
    // darüber GLOBAL (mit Reuse). Fallback ist immer GLOBAL.
    inline constexpr int    perturbUseConstUpTo   = 8192;   // 4K..16K

    // Guard-Faktor K: zulässiges |δ| ≤ K * pxScale (Screen-Pixel-Maß).
    inline constexpr double perturbDeltaGuardK    = 6.0;    // 4.0..10.0

    // Rebase-Schwelle in Pixeln (Screen Space): bei Δview ≥ RebaseDeltaPx → Rebase.
    inline constexpr double perturbRebaseDeltaPx  = 1.25;   // 0.75..2.0

} // namespace Settings
