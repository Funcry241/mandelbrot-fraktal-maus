#pragma once

// settings.hpp — 🐭 Alle zentralen Konstanten kompakt & verständlich kommentiert

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf

namespace Settings {

// 🔧 Debugging / Test-Modus
inline constexpr bool debugGradient = false;   // Testweise Farbverlauf anstelle von Mandelbrot anzeigen
inline constexpr bool debugLogging  = true;    // Detaillierte Debug-Ausgaben aktivieren

// 💻 Fenster- und Bildkonfiguration
inline constexpr int width        = 1024;      // Fensterbreite in Pixeln
inline constexpr int height       = 768;       // Fensterhöhe in Pixeln
inline constexpr int windowPosX   = 100;       // Initiale Fensterposition X
inline constexpr int windowPosY   = 100;       // Initiale Fensterposition Y

// 🔎 Zoom- und Navigationsparameter
inline constexpr float initialZoom    = 300.0f;  // Start-Zoomstufe — höher = tieferer Einstieg
inline constexpr float zoomFactor     = 1.01f;   // Faktor pro Zoom-Schritt (>1.0 -> Vergrößerung)
inline constexpr float initialOffsetX = -0.5f;   // Start-Offset auf der X-Achse
inline constexpr float initialOffsetY =  0.0f;   // Start-Offset auf der Y-Achse

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;     // Faktor für Offset-Änderungen bei Verschieben
inline constexpr float ZOOM_STEP_FACTOR   = 0.002f;    // Faktor für sanfte Zoom-Schritte (je Frame)

inline constexpr float MIN_OFFSET_STEP = 1e-8f;        // Minimale Schrittweite beim Verschieben
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;        // Minimale Änderung beim Zoomen

// 🧐 Auto-Zoom Steuerung — Variance
inline constexpr float VARIANCE_THRESHOLD      = 1e-12f;  // Ausgangs-Schwelle für interessante Bildbereiche
inline constexpr float MIN_VARIANCE_THRESHOLD  = 1e-10f;  // Verhindert, dass die Schwelle zu klein wird (sonst Blindflug)

// Dynamischer Variance-Threshold in Abhängigkeit vom Zoom
inline float dynamicVarianceThreshold(float zoom) {
    return std::max(VARIANCE_THRESHOLD / logf(zoom + 2.0f), MIN_VARIANCE_THRESHOLD);
}

// 🔎 Auto-Zoom Steuerung — Suchradius
inline constexpr float DYNAMIC_RADIUS_SCALE = 0.05f; // Skaliert den Suchradius basierend auf √Zoom
inline constexpr int   DYNAMIC_RADIUS_MIN   = 20;    // Minimaler Suchradius in Tiles
inline constexpr int   DYNAMIC_RADIUS_MAX   = 300;   // Maximaler Suchradius in Tiles

// 🔢 Iterationsparameter
inline constexpr int INITIAL_ITERATIONS = 100;   // Startwert für Iterationen
inline constexpr int MAX_ITERATIONS_CAP = 5000;  // Obergrenze für Iterationen
inline constexpr int ITERATION_STEP     = 5;     // Erhöhungsschritte bei Progressiv-Rendern

// 🐾 Sanftes Gliding für Offset-Animationen
inline constexpr float LERP_FACTOR = 0.02f;      // Geschwindigkeit der Zielanpassung (kleiner = weicher)

// 🧩 Dynamische Tile-Größe (adaptive Kacheln)
inline constexpr int BASE_TILE_SIZE = 8;     // Basisgröße für Tiles
inline constexpr int MIN_TILE_SIZE  = 4;     // Minimale Tile-Größe
inline constexpr int MAX_TILE_SIZE  = 32;    // Maximale Tile-Größe

// 🧩 Feste Tile-Größen für statische CUDA-Grid-Berechnung
inline constexpr int TILE_W = 16;            // CUDA Blockbreite
inline constexpr int TILE_H = 16;            // CUDA Blockhöhe

// 🚀 Adaptive Tile-Berechnung basierend auf dem Zoom
inline int dynamicTileSize(float zoom) {
    static int lastSize = -1;  // 🐭 Merkt sich letzte TileSize

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
#if defined(DEBUG) || defined(_DEBUG) || Settings::debugLogging
        std::printf("[DEBUG] TileSize changed to %d\n", bestSize);
#endif
        lastSize = bestSize;
    }

    return bestSize;
}


} // namespace Settings
