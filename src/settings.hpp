#pragma once

// Datei: src/settings.hpp
// Zeilen: 103
// 🐭 Maus-Kommentar: Steuerungszentrale für Zoomlogik, Fraktal-Feintuning, Entropie-Autoanalyse, Loggingsteuerung und CUDA-Tile-Verhalten.
// Variablen sind so benannt, dass auch Schwester sofort weiß, was sie tun. Keine magischen Zahlen mehr. Schneefuchs hätte diese Dokumentation geliebt.

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf, log2f, sqrtf

namespace Settings {

// 🔍 Debug-Modi: visuelle Darstellung & Konsolen-Ausgabe
inline constexpr bool debugGradient = false; // Zeige Gradient-Vorschau statt Farben
inline constexpr bool debugLogging  = true;  // Aktiviere DEBUG-Ausgaben in Konsole (z. B. bei Zoomwechseln)

// 🖥️ Fenstergröße & -position
inline constexpr int width        = 1024;  // Fensterbreite in Pixel
inline constexpr int height       = 768;   // Fensterhöhe in Pixel
inline constexpr int windowPosX   = 100;   // Startposition X
inline constexpr int windowPosY   = 100;   // Startposition Y

// 🔭 Anfangszustand für Zoom und Fraktalposition
inline constexpr float initialZoom    = 300.0f;  // Anfangszoom
inline constexpr float initialOffsetX = -0.5f;   // X-Verschiebung (Start im Mandelbrot-Set)
inline constexpr float initialOffsetY =  0.0f;   // Y-Verschiebung

inline constexpr float zoomFactor = 1.005f;      // Wie stark gezoomt wird pro Frame
inline constexpr float lerpFactor = 0.008f;       // Interpolationsfaktor für Offset-Anpassung

// 🔍 Manueller Zoom per Mausrad oder Tastatur
inline constexpr float ZOOM_STEP_FACTOR = 0.002f; // Zoomänderung pro Scrollschritt

// 🎯 Entropie-Schwelle für Auto-Zoom-Entscheidung
inline constexpr float VARIANCE_THRESHOLD     = 1e-12f; // Standard-Schwelle (Startwert)
inline constexpr float MIN_VARIANCE_THRESHOLD = 1e-10f; // Harte Untergrenze

// 🌀 Auto-Zoom Geschwindigkeit: größer = schnelleres Hineinzoomen
inline constexpr float AUTOZOOM_SPEED = 1.01f; // Faktor für schrittweisen Zoomanstieg

// 🔁 Iterationsverhalten: Fraktal-Schärfe & Performance
inline constexpr int INITIAL_ITERATIONS = 100;   // Startanzahl Iterationen
inline constexpr int MAX_ITERATIONS_CAP = 5000;  // Obergrenze (zur Sicherheit)
inline constexpr int ITERATION_STEP     = 5;     // Schrittgröße bei Anpassung

// 🧲 Sanfte Bewegung beim Auto-Zoom (TileCenter → Offset)
inline constexpr float LERP_FACTOR = 0.02f; // Interpolationsfaktor – 0.0 = kein Zoomsprung, 1.0 = harter Sprung

// 🚫 Mindestdistanz für Offset-Änderung (verhindert "Zoomzittern")
inline constexpr float MIN_JUMP_DISTANCE = 1e-4f; // Verhindert Bewegung, wenn TileCenter ≈ Offset

// 🔲 Tile-Größen (für CUDA-Aufteilung & Entropieanalyse)
inline constexpr int BASE_TILE_SIZE = 8;  // Richtgröße vor Berechnung
inline constexpr int MIN_TILE_SIZE  = 4;  // Untergrenze
inline constexpr int MAX_TILE_SIZE  = 32; // Obergrenze

// 📐 Zusätzliche Tile-Maße für HUD oder Grid-Overlays (optional)
inline constexpr int TILE_W = 16;
inline constexpr int TILE_H = 16;

// 📏 Dynamische Tile-Größe abhängig vom Zoom-Level
inline int dynamicTileSize(float zoom) {
    static int lastSize = -1;

    float logZoom = log10f(zoom + 1.0f);
    float rawSize = BASE_TILE_SIZE * (8.0f / (logZoom + 1.0f));

    constexpr int allowedSizes[] = {32, 16, 8, 4};

    int bestSize = allowedSizes[0];
    for (int size : allowedSizes) {
        if (rawSize >= size) {
            bestSize = size;
            break;
        }
    }

    if (bestSize != lastSize) {
#if defined(DEBUG) || defined(_DEBUG)
        if (Settings::debugLogging) {
            std::printf("[DEBUG] TileSize changed to %d\n", bestSize);
        }
#endif
        lastSize = bestSize;
    }

    return bestSize;
}

// 📉 Skaliere VARIANCE_THRESHOLD mit dem Zoom-Level (für adaptive Empfindlichkeit)
inline float dynamicVarianceThreshold(float zoom) {
    float scaled = VARIANCE_THRESHOLD * (1.0f + 0.02f * log2f(zoom + 1.0f));
    return std::clamp(scaled, VARIANCE_THRESHOLD, MIN_VARIANCE_THRESHOLD * 10.0f);
}

} // namespace Settings
