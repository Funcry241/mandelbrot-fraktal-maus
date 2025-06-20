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
inline constexpr float LERP_FACTOR = 0.075f;      // sanftes Nachziehen
inline constexpr float DEADZONE    = 1e-9f;       // Bild bleibt ruhig, wenn Ziel erreicht

// 📈 Gewichtung für Entropie-Nähe-Bonus im Auto-Zoom
inline constexpr float ENTROPY_NEARBY_BIAS = 0.5f;

// 💚 CUDA-Tile-Einstellungen
inline constexpr int BASE_TILE_SIZE = 8;

// 🐭 Maus-Kommentar: Eigene clamp-Funktion, um Konflikte mit <algorithm> (std::clamp) und PCH zu vermeiden.
// Diese Funktion begrenzt einen Wert `val` auf das Intervall [minVal, maxVal].
// Schneefuchs meinte: „Immer schön im Rahmen bleiben – wie ein Otter im Bau!“
inline float my_clamp(float val, float minVal, float maxVal) {
    return (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
}

} // namespace Settings
