#pragma once

// Datei: src/settings.hpp
// Zeilen: 83
// 🐅 Maus-Kommentar: Steuerungszentrale für Auto-Zoom, Fraktal-Feintuning, Entropieanalyse und CUDA-Tile-Verhalten.
// Nur aktive, genutzte Konstanten bleiben erhalten – der Rest wurde entrümpelt. Die Schwester atmet auf.

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf, log2f, sqrtf

namespace Settings {

// 🔍 Debug-Modi: visuelle Darstellung & Konsolen-Ausgabe aktivieren
inline constexpr bool debugGradient = false; // Zeige nur den Entropie-Gradienten (statt Farben)
inline constexpr bool debugLogging  = true;  // Zusätzliche Debug-Ausgaben im Terminal anzeigen

// 🖥️ Fensterkonfiguration (Initialgröße und Position auf dem Bildschirm)
inline constexpr int width        = 1024;
inline constexpr int height       = 768;
inline constexpr int windowPosX   = 100;
inline constexpr int windowPosY   = 100;

// 🔭 Initialer Fraktal-Ausschnitt (Zoom und Position)
inline constexpr float initialZoom    = 300.0f;
inline constexpr float initialOffsetX = -0.5f;
inline constexpr float initialOffsetY =  0.0f;

// 🔍 Manueller Zoom (z. B. per Mausrad) pro Schritt
inline constexpr float ZOOM_STEP_FACTOR = 0.002f;

// 🌟 Schwelle zur Erkennung "interessanter" Tiles via Entropie
inline constexpr float VARIANCE_THRESHOLD     = 1e-12f;
inline constexpr float MIN_VARIANCE_THRESHOLD = 1e-10f;

// 🌀 Wie schnell zoomt das Bild automatisch pro Frame
inline constexpr float AUTOZOOM_SPEED = 1.005f;

// ♻️ Steuerung der Fraktaldarstellung durch Iterationsanzahl
inline constexpr int INITIAL_ITERATIONS = 100;
inline constexpr int MAX_ITERATIONS_CAP = 50000;
inline constexpr int ITERATION_STEP     = 5;

// 🪞 Glättung der Kamerabewegung zum Ziel-Tile
inline constexpr float LERP_FACTOR = 0.02f;

// 📈 Gewichtung für Entropie-Nähe-Bonus im Auto-Zoom
inline constexpr float ENTROPY_NEARBY_BIAS = 60.0f;

// 💚 CUDA-Tile-Einstellungen
inline constexpr int BASE_TILE_SIZE = 8;

// 📊 Tile-Größe passt sich dynamisch dem Zoom-Level an
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

// 📈 Entropie-Schwelle passt sich dem Zoomlevel an
inline float dynamicVarianceThreshold(float zoom) {
    float scaled = VARIANCE_THRESHOLD * (1.0f + 0.02f * log2f(zoom + 1.0f));
    return std::clamp(scaled, VARIANCE_THRESHOLD, MIN_VARIANCE_THRESHOLD * 10.0f);
}

} // namespace Settings
