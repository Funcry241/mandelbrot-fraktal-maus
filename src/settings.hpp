// MAUS:
// ============================================================================
// Datei: src/settings.hpp
// Central project settings — fully documented.
// Policy: All runtime LOG/DEBUG output must be English and ASCII-only.
// Comments here may be German. References:
//   – Otter = vom User inspiriert (Bezug zu Otter)
//   – Schneefuchs = von Schwester inspiriert (Bezug zu Schneefuchs)
// Keine Regressionen, keine versteckten Semantik-Änderungen. Werte bleiben identisch.
// ============================================================================

#pragma once

namespace Settings {

    // ------------------------------------------------------------------------
    // ForceAlwaysZoom
    // Wirkung:
    //   Erzwingt kontinuierliches Zoomen, unabhängig von Entropie/Kontrast-
    //   Ergebnissen. Dient als „Drift“-Fallback, damit die Kamera stets Bewegung
    //   zeigt, selbst wenn die Analyse mal kein klares Ziel liefert.
    //
    // Empfehlung (Min..Max):
    //   false .. true  (bool)
    //
    // Erhöhung/Reduzierung:
    //   • true (Erhöhung auf aktiv): Stetige Bewegung, ideal für Demos und
    //     zur Vermeidung von Stagnation bei flachen Regionen. (Bezug zu Otter)
    //   • false (Reduzierung auf aus): Bewegung ergibt sich nur aus den
    //     Analysewerten; kann „ehrlicher“, aber auch stoppanfälliger sein.
    //
    // Hinweis:
    //   Aktiv lassen, wenn „Silk‑Lite“ stets flüssig wirken soll; deaktivieren,
    //   um die Treffgüte der Zielsuche zu verifizieren. (Bezug zu Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr bool   ForceAlwaysZoom = true;   // 🦦 Otter: „always zoom“

    // ------------------------------------------------------------------------
    // ForcedZoomStep
    // Wirkung:
    //   Multiplikativer Zoomfaktor pro Frame, wenn ForceAlwaysZoom aktiv ist.
    //   Werte < 1.0 zoomen rein, Werte > 1.0 zoomen raus.
    //
    // Empfehlung (Min..Max):
    //   0.90 .. 0.999 (double)
    //   Typisch für sanften Drift: 0.95 .. 0.985
    //
    // Erhöhung/Reduzierung:
    //   • Näher an 1.0 (Erhöhung): Langsamer, subtiler Zoom; ruhiger Look.
    //   • Richtung 0.90 (Reduzierung): Schneller Zoom; eindrucksvoll, aber
    //     potenziell unruhig und iterationsintensiver. (Bezug zu Otter)
    //
    // Hinweis:
    //   PD‑Motion‑Planner kann die visuelle Geschwindigkeit zusätzlich glätten.
    // ------------------------------------------------------------------------
    inline constexpr double ForcedZoomStep  = 0.97;   // 🦦 Otter: smooth & steady

    // ------------------------------------------------------------------------
    // debugLogging
    // Wirkung:
    //   Schaltet gezielte Debug-/Diagnose-Ausgaben im Host/Device‑Pfad frei
    //   (Timing, Kernel‑Phasen, Indizes, etc.). Laufzeitlogs sind stets EN/ASCII.
    //
    // Empfehlung (Min..Max):
    //   false .. true  (bool)
    //
    // Erhöhung/Reduzierung:
    //   • true (Erhöhung auf aktiv): Mehr Einblick, evtl. leichte FPS‑Kosten.
    //   • false (Reduzierung auf aus): Ruhiger Lauf, ideal für Captures/Bench.
    //
    // Hinweis:
    //   Nur aktivieren, wenn du reproduzierbare Fragen klären willst; sonst aus.
    //   Deterministische, sparsame Device‑Logs bevorzugen. (Bezug zu Schneefuchs)
    // ------------------------------------------------------------------------
    constexpr bool debugLogging  = true;

    // ------------------------------------------------------------------------
    // heatmapOverlayEnabled
    // Wirkung:
    //   Bestimmt, ob das Heatmap‑Overlay beim Programmstart sichtbar ist.
    //
    // Empfehlung (Min..Max):
    //   false .. true  (bool)
    //
    // Erhöhung/Reduzierung:
    //   • true: Sofortige Diagnose/Analyse im Bild; gut für Entwicklung.
    //   • false: Aufgeräumteres Bild; Overlay bei Bedarf zuschalten.
    //
    // Hinweis:
    //   Für „Pfote/Eule“‑Diagnosen sinnvoll zunächst aktiv. (Bezug zu Otter)
    // ------------------------------------------------------------------------
    constexpr bool heatmapOverlayEnabled = true; 

    // ------------------------------------------------------------------------
    // warzenschweinOverlayEnabled
    // Wirkung:
    //   Schaltet das Warzenschwein‑HUD (FPS/Stats/Text) an/aus zum Start.
    //
    // Empfehlung (Min..Max):
    //   false .. true  (bool)
    //
    // Erhöhung/Reduzierung:
    //   • true: HUD sofort sichtbar (nützlich beim Tuning).
    //   • false: Clean Look; HUD nur auf Anfrage.
    //
    // Hinweis:
    //   Für „WOW‑Effekt“‑Style‑Checks zu Beginn aktiv lassen. (Bezug zu Otter)
    // ------------------------------------------------------------------------
    constexpr bool warzenschweinOverlayEnabled = true; 

    // ------------------------------------------------------------------------
    // hudPixelSize
    // Wirkung:
    //   Skalierung der HUD‑Glyphen in NDC‑Einheiten pro Pixelquadrat.
    //   Steuert die wahrgenommene Textgröße.
    //
    // Empfehlung (Min..Max):
    //   0.0015f .. 0.0040f
    //
    // Erhöhung/Reduzierung:
    //   • Höher: Größerer, präsenterer Text (Risiko: Clipping/Überlagerung).
    //   • Niedriger: Dezenter; kann auf hochauflösenden Displays zu klein sein.
    //
    // Hinweis:
    //   An DPI/Fenstergröße koppeln, wenn dynamisch benötigt. (Bezug zu Schneefuchs)
    // ------------------------------------------------------------------------
    inline constexpr float hudPixelSize = 0.0025f;

    // ------------------------------------------------------------------------
    // Fensterkonfiguration (width/height/windowPosX/windowPosY)
    // Wirkung:
    //   Startgröße und Startposition des Fensters.
    //
    // Empfehlung (Min..Max):
    //   width :  800 .. 3840
    //   height:  600 .. 2160
    //   Pos   : frei (je nach Multi‑Monitor‑Setup)
    //
    // Erhöhung/Reduzierung:
    //   • Größere Fenster: Mehr Pixel → höhere GPU‑Last, klareres Bild.
    //   • Kleinere Fenster: Höhere FPS, geringere Renderlast.
    //
    // Hinweis:
    //   Für reproduzierbare Benchmarks feste Startwerte beibehalten. (Schneefuchs)
    // ------------------------------------------------------------------------
    constexpr int width       = 1024;
    constexpr int height      = 768;
    constexpr int windowPosX  = 100;
    constexpr int windowPosY  = 100;

    // ------------------------------------------------------------------------
    // Initialer Fraktal‑Ausschnitt (initialZoom, initialOffsetX/Y)
    // Wirkung:
    //   Start‑Zoom und ‑Offset im Komplexraum. Bestimmt den Erst‑Eindruck.
    //
    // Empfehlung (Min..Max):
    //   initialZoom   : 0.5f .. 10.0f
    //   initialOffsetX: -2.0f .. +2.0f
    //   initialOffsetY: -2.0f .. +2.0f
    //
    // Erhöhung/Reduzierung:
    //   • Höherer initialZoom: Näher dran, mehr Detail, höhere Iterationskosten.
    //   • Niedrigerer initialZoom: Weiterer Blick, schneller, weniger Details.
    //
    // Hinweis:
    //   Aktuelle Defaults liefern einen klassischen Mandelbrot‑Frame. (Otter)
    // ------------------------------------------------------------------------
    constexpr float initialZoom    = 1.5f;
    constexpr float initialOffsetX = -0.5f;
    constexpr float initialOffsetY = 0.0f;

    // ------------------------------------------------------------------------
    // Iterationssteuerung (INITIAL_ITERATIONS, MAX_ITERATIONS_CAP)
    // Wirkung:
    //   INITIAL_ITERATIONS: Startbudget der Iterationen pro Pixel.
    //   MAX_ITERATIONS_CAP: Harte Obergrenze gegen Explodieren der Kosten.
    //
    // Empfehlung (Min..Max):
    //   INITIAL_ITERATIONS: 64 .. 1024
    //   MAX_ITERATIONS_CAP: 4096 .. 200000 (GPU/Anspruch abhängig)
    //
    // Erhöhung/Reduzierung:
    //   • INITIAL_ITERATIONS ↑: Sauberere Kanten, mehr Kosten/FPS‑Verlust möglich.
    //   • MAX_ITERATIONS_CAP ↑: Tiefere Zoomschärfe; Risiko von Frame‑Spikes.
    //   • ↓ jeweils entsprechend entlastend, aber gröber.
    //
    // Hinweis:
    //   Bei aggressivem Zoom (ForcedZoomStep klein) Obergrenzen im Blick behalten.
    //   Dokumentationspflicht: Warnung vor Perf‑Cliffs. (Bezug zu Schneefuchs)
    // ------------------------------------------------------------------------
    constexpr int INITIAL_ITERATIONS = 100;
    constexpr int MAX_ITERATIONS_CAP = 50000;

    // ------------------------------------------------------------------------
    // CUDA Tile‑Größen (BASE/MIN/MAX_TILE_SIZE)
    // Wirkung:
    //   Steuern die Arbeitsaufteilung in Kacheln für CUDA‑Kernels.
    //   Kachelkanten sollten i. d. R. vielfache von 8 sein (Warp‑freundlich).
    //
    // Empfehlung (Min..Max):
    //   BASE_TILE_SIZE: 16 .. 64
    //   MIN_TILE_SIZE :  8 .. BASE_TILE_SIZE
    //   MAX_TILE_SIZE : BASE_TILE_SIZE .. 128
    //
    // Erhöhung/Reduzierung:
    //   • Größere Tiles: Weniger Launches, potenziell bessere Coalescing‑Effekte,
    //     aber ungleichmäßige Lastverteilung bei komplexen Regionen möglich.
    //   • Kleinere Tiles: Feinere Verteilung, mehr Launch‑Overhead/Synchronisation.
    //
    // Hinweis:
    //   In Einklang mit Zoom‑Geschwindigkeit und FPS‑Zielen tunen. (Bezug zu Otter)
    // ------------------------------------------------------------------------
    constexpr int BASE_TILE_SIZE = 32;
    constexpr int MIN_TILE_SIZE  = 8;
    constexpr int MAX_TILE_SIZE  = 64;

} // namespace Settings
