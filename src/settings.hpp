#pragma once

// settings.hpp — Alle zentralen Konstanten kompakt & verständlich kommentiert

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf, log2f, sqrtf

namespace Settings {

inline constexpr bool debugGradient = false;   // Testweise Farbverlauf anstelle von Mandelbrot anzeigen
inline constexpr bool debugLogging  = true;    // Detaillierte Debug-Ausgaben aktivieren

inline constexpr int width        = 1024;      // Fensterbreite in Pixeln
inline constexpr int height       = 768;       // Fensterhöhe in Pixeln
inline constexpr int windowPosX   = 100;       // Initiale Fensterposition X
inline constexpr int windowPosY   = 100;       // Initiale Fensterposition Y

inline constexpr float initialZoom    = 300.0f;  // Start-Zoomstufe — höher = tieferer Einstieg
inline constexpr float zoomFactor     = 1.01f;   // Faktor pro Zoom-Schritt (>1.0 -> Vergrößerung)
inline constexpr float initialOffsetX = -0.5f;   // Start-Offset auf der X-Achse
inline constexpr float initialOffsetY =  0.0f;   // Start-Offset auf der Y-Achse

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;     // Faktor für Offset-Änderungen bei Verschieben
inline constexpr float ZOOM_STEP_FACTOR   = 0.002f;    // Faktor für sanfte Zoom-Schritte (je Frame)

inline constexpr float MIN_OFFSET_STEP = 1e-8f;        // Minimale Schrittweite beim Verschieben
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;        // Minimale Änderung beim Zoomen

inline constexpr float VARIANCE_THRESHOLD      = 1e-12f;  // Ausgangs-Schwelle für interessante Bildbereiche
inline constexpr float MIN_VARIANCE_THRESHOLD  = 1e-10f;  // Verhindert, dass die Schwelle zu klein wird (sonst Blindflug)

inline constexpr float AUTOZOOM_SPEED = 1.01f;  // Zoom-Faktor bei Auto-Zoom

inline constexpr float DYNAMIC_RADIUS_SCALE = 1.0f;   // Skaliert den Suchradius basierend auf √Zoom
inline constexpr int   DYNAMIC_RADIUS_MIN   = 20;     // Minimaler Suchradius in Tiles
inline constexpr int   DYNAMIC_RADIUS_MAX   = 300;    // Maximaler Suchradius in Tiles

inline constexpr int INITIAL_ITERATIONS = 100;   // Startwert für Iterationen
inline constexpr int MAX_ITERATIONS_CAP = 5000;  // Obergrenze für Iterationen
inline constexpr int ITERATION_STEP     = 5;     // Erhöhungsschritte bei Progressiv-Rendern

inline constexpr float LERP_FACTOR = 0.02f;      // Geschwindigkeit der Zielanpassung (kleiner = weicher)

inline constexpr int BASE_TILE_SIZE = 8;     // Basisgröße für Tiles
inline constexpr int MIN_TILE_SIZE  = 4;     // Minimale Tile-Größe
inline constexpr int MAX_TILE_SIZE  = 32;    // Maximale Tile-Größe

inline constexpr int TILE_W = 16;            // CUDA Blockbreite
inline constexpr int TILE_H = 16;            // CUDA Blockhöhe

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

inline float dynamicVarianceThreshold(float zoom) {
    float scaled = VARIANCE_THRESHOLD * (1.0f + 0.02f * log2f(zoom + 1.0f));
    return std::clamp(scaled, VARIANCE_THRESHOLD, MIN_VARIANCE_THRESHOLD * 10.0f);
}

inline int dynamicSearchRadius(float zoom) {
    float radius = DYNAMIC_RADIUS_SCALE * sqrtf(zoom);
    return std::clamp(static_cast<int>(radius), DYNAMIC_RADIUS_MIN, DYNAMIC_RADIUS_MAX);
}

inline int dynamicIterationLimit(float zoom) {
    float boost = 1.0f + 0.001f * zoom;
    int iterations = static_cast<int>(INITIAL_ITERATIONS * boost);
    return std::min(iterations, MAX_ITERATIONS_CAP);
}

} // namespace Settings
