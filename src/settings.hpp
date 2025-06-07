#pragma once

// settings.hpp — 🐭 Alle zentralen Konstanten kompakt & modern gesammelt

namespace Settings {

// ----------------------------------------------------------------------
// 🛠️ Debugging / Test-Modus
inline constexpr bool debugGradient = false;    // Testbild-Modus aktivieren
inline constexpr bool debugLogging  = false;    // Viel Konsolenausgabe (Debug-Log)

// ----------------------------------------------------------------------
// 🖥️ Fenster und Bild
inline constexpr int width        = 1024;       // Fensterbreite
inline constexpr int height       = 768;        // Fensterhöhe
inline constexpr int windowPosX   = 100;        // Fenster-Startposition X
inline constexpr int windowPosY   = 100;        // Fenster-Startposition Y

// ----------------------------------------------------------------------
// 🔎 Zoom & Pan Einstellungen
inline constexpr float initialZoom    = 300.0f;  // Anfangszoom
inline constexpr float zoomFactor     = 1.01f;   // Zoom-Multiplikator pro Schritt
inline constexpr float initialOffsetX = -0.5f;   // Startversatz X
inline constexpr float initialOffsetY =  0.0f;   // Startversatz Y

// Zoom- und Pan-Steuerung (dynamisch zur Zoomstufe angepasst)
inline constexpr float OFFSET_STEP_FACTOR = 0.5f;     // Basis-Offset pro Frame (skaliert mit 1/Zoom)
inline constexpr float ZOOM_STEP_FACTOR   = 0.15f;    // Basis-Zoomrate pro Frame

inline constexpr float MIN_OFFSET_STEP = 1e-8f;       // Minimal erlaubter Pan-Schritt
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;       // Minimal erlaubter Zoom-Schritt

// ----------------------------------------------------------------------
// 🧠 Auto-Zoom Steuerung
inline constexpr float VARIANCE_THRESHOLD = 1e-12f;   // Basis-Schwelle für Varianz

// Dynamischer Variance-Threshold in Abhängigkeit vom Zoom
inline float dynamicVarianceThreshold(float zoom) {
    // Maus-Kommentar: logarithmische Abhängigkeit für feinfühlige Schwelle
    return VARIANCE_THRESHOLD / logf(zoom + 2.0f);
}

// ----------------------------------------------------------------------
// 🔢 Iterations-Steuerung
inline constexpr int TILE_W             = 16;    // Kachelbreite für CUDA-Block
inline constexpr int TILE_H             = 16;    // Kachelhöhe für CUDA-Block
inline constexpr int INITIAL_ITERATIONS = 100;   // Startwert für progressive Iterationen
inline constexpr int MAX_ITERATIONS_CAP = 5000;  // Obergrenze für Iterationen
inline constexpr int ITERATION_STEP     = 5;     // Iterationszuwachs pro Frame

} // namespace Settings
