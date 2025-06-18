#pragma once

// Datei: src/settings.hpp
// Zeilen: 93
// 🐅 Maus-Kommentar: Steuerungszentrale für Auto-Zoom, Fraktal-Feintuning, Entropieanalyse und CUDA-Tile-Verhalten.
// MIN_JUMP_DISTANCE wurde deaktiviert – Zoom läuft jetzt dauerhaft, LERP bleibt. Schwester kann jetzt mit gutem Gewissen loslassen.

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf, log2f, sqrtf

namespace Settings {

// 🔍 Debug-Modi: visuelle Darstellung & Konsolen-Ausgabe aktivieren
inline constexpr bool debugGradient = false; // Zeige nur den Entropie-Gradienten (statt Farben)
inline constexpr bool debugLogging  = true;  // Zusätzliche Debug-Ausgaben im Terminal anzeigen

// 🖥️ Fensterkonfiguration (Initialgröße und Position auf dem Bildschirm)
inline constexpr int width        = 1024;  // Fensterbreite in Pixel
inline constexpr int height       = 768;   // Fensterhöhe in Pixel
inline constexpr int windowPosX   = 100;   // X-Startposition
inline constexpr int windowPosY   = 100;   // Y-Startposition

// 🔭 Initialer Fraktal-Ausschnitt (Zoom und Position)
inline constexpr float initialZoom    = 300.0f; // Anfangszoom-Stufe (Skalierungsfaktor)
inline constexpr float initialOffsetX = -0.5f;  // Startverschiebung X-Achse
inline constexpr float initialOffsetY =  0.0f;  // Startverschiebung Y-Achse

// 🔍 Manueller Zoom (z. B. per Mausrad) pro Schritt
inline constexpr float ZOOM_STEP_FACTOR = 0.002f; // Kleinere Werte = feinere Zoomkontrolle

// 🌟 Schwelle zur Erkennung "interessanter" Tiles via Entropie
inline constexpr float VARIANCE_THRESHOLD     = 1e-12f; // Standard-Sensitivität für Tile-Komplexität
inline constexpr float MIN_VARIANCE_THRESHOLD = 1e-10f; // Untergrenze der Schwelle

// 🌀 Wie schnell zoomt das Bild automatisch pro Frame
inline constexpr float AUTOZOOM_SPEED = 1.005f; // Jeder Frame: zoom *= AUTOZOOM_SPEED

// ♻️ Steuerung der Fraktaldarstellung durch Iterationsanzahl
inline constexpr int INITIAL_ITERATIONS = 100;     // Startwert für Iterationen
inline constexpr int MAX_ITERATIONS_CAP = 50000;   // Harte Obergrenze für Qualität / Performance
inline constexpr int ITERATION_STEP     = 5;       // Schrittweite bei Progression

// 🪞 Glättung der Kamerabewegung zum Ziel-Tile (statt harten Sprung)
inline constexpr float LERP_FACTOR = 0.02f; // Zwischen 0.0 (sanft) und 1.0 (sofort)

// ❌ Mindestdistanz für Bewegung (nicht mehr aktiv genutzt)
// inline constexpr float MIN_JUMP_DISTANCE = 1e-4f;

// 📈 Gewichtung für Entropie-Nähe-Bonus im Auto-Zoom (je höher, desto stärker der Nahbereich bevorzugt)
inline constexpr float ENTROPY_NEARBY_BIAS = 60.0f;

// 💚 CUDA-Tile-Einstellungen (wichtig für Parallelisierung & Analyse)
inline constexpr int BASE_TILE_SIZE = 8;
inline constexpr int MIN_TILE_SIZE  = 4;
inline constexpr int MAX_TILE_SIZE  = 32;

// 📏 Feste Tile-Maße (optional für Grid-Overlays oder Debug-Darstellung)
inline constexpr int TILE_W = 16;
inline constexpr int TILE_H = 16;

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

// 📈 Variance-Schwelle wird mit Zoom mitskaliert (empfindlicher bei großem Zoom)
inline float dynamicVarianceThreshold(float zoom) {
    float scaled = VARIANCE_THRESHOLD * (1.0f + 0.02f * log2f(zoom + 1.0f));
    return std::clamp(scaled, VARIANCE_THRESHOLD, MIN_VARIANCE_THRESHOLD * 10.0f);
}

} // namespace Settings
