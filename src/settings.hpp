#pragma once

// Datei: src/settings.hpp
// Zeilen: 94
// 🐅 Maus-Kommentar: Steuerungszentrale für Auto-Zoom, Fraktal-Feintuning, Entropieanalyse und CUDA-Tile-Verhalten.
// Nur aktive, genutzte Konstanten bleiben erhalten – der Rest wurde entrümpelt. Die Schwester atmet auf.

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf, log2f, sqrtf

namespace Settings {

// 🔍 Debug-Modi: visuelle Darstellung & Konsolen-Ausgabe aktivieren
inline constexpr bool debugGradient = false; // Aktiviert reine Entropie-Ansicht (keine Farben) – nur zu Analysezwecken
inline constexpr bool debugLogging  = true;  // Aktiviert Konsolenausgaben für Auto-Zoom, Tile-Entropie etc.

// 💥 Fensterkonfiguration (Initialgröße und Position auf dem Bildschirm)
inline constexpr int width      = 1024;  // Breite des Fensters in Pixel – empfohlen: 800 bis 1600
inline constexpr int height     = 768;   // Höhe des Fensters in Pixel – empfohlen: 600 bis 1200
inline constexpr int windowPosX = 100;   // Startposition links
inline constexpr int windowPosY = 100;   // Startposition oben

// 🔭 Initialer Fraktal-Ausschnitt (Zoomfaktor und Verschiebung)
inline constexpr float initialZoom    = 300.0f;   // Start-Zoom: höherer Wert = näher dran – empfohlen: 100–1000
inline constexpr float initialOffsetX = -0.5f;    // Anfangsverschiebung auf der X-Achse
inline constexpr float initialOffsetY =  0.0f;    // Anfangsverschiebung auf der Y-Achse

// 🔍 Manueller Zoom (per Mausrad oder Tasten) – pro Schritt
inline constexpr float ZOOM_STEP_FACTOR = 0.002f; // Erhöhung = schnelleres Zoomen – empfohlen: 0.001 bis 0.01

// 🌟 Schwellenwerte für Entropieanalyse zur Auswahl interessanter Tiles
inline constexpr float VARIANCE_THRESHOLD     = 0.01f; // Hauptschwelle für interessante Tiles – je kleiner, desto empfindlicher
inline constexpr float MIN_VARIANCE_THRESHOLD = 0.001f; // Notbremse für zu starkes Auto-Zoom – empfohlen: 1e-10 bis 1e-8

// 🌀 Geschwindigkeit des automatischen Zooms pro Frame
inline constexpr float AUTOZOOM_SPEED = 1.005f; // Höher = schnellerer Zoom – empfohlen: 1.002 bis 1.01

// 🪎 Glättungsfaktor für Kamera-Nachführung zum Ziel (linearer LERP)
// Kleiner = langsameres Nachziehen, größer = schneller & unruhiger
inline constexpr float LERP_FACTOR = 0.035f;  // empfohlen: 0.01 bis 0.08

// 🦕 Stillstandsschwelle für Offset-Bewegung – wirkt wie ein Ruhepuffer
// Wenn Offset näher als DEADZONE am Ziel liegt, wird keine Bewegung mehr ausgeführt
inline constexpr float DEADZONE = 1e-8f;  // empfohlen: 1e-10 bis 1e-8 – kleiner = empfindlicher

// 🦕 Maximaler Anteil der Ziel-Distanz, der pro Frame bewegt werden darf (in Fraktal-Koordinaten)
// Limitiert Bewegungsgeschwindigkeit zusätzlich zur tanh-Dämpfung
inline constexpr float MAX_OFFSET_FRACTION = 0.1f; // empfohlen: 0.05 bis 0.2 – größer = schnelleres Nachziehen

// 📈 Bonusgewichtung für Tiles, die nah am aktuellen Offset liegen (für stabileres Auto-Zoom)
// 0.0 = keine Bevorzugung, 1.0 = starker Bias auf Nähe
inline constexpr float ENTROPY_NEARBY_BIAS = 0.5f; // empfohlen: 0.0 bis 0.6 – höher = weniger Hüpfen

// 🦕 Skaliert die Offset-Distanz vor Anwendung von tanh (nonlineare Dämpfung)
// Kleinere Werte = stärkere Dämpfung bei kleinen Bewegungen
// Empfohlen: 1.0 bis 10.0 – z.‌B. 5.0 bedeutet, dass bei tanh(5.0 * distance) ≈ 1 schnelle Bewegung erfolgt
inline constexpr float OFFSET_TANH_SCALE = 5.0f;

// ♻️ Iterationssteuerung – beeinflusst Detailtiefe bei starkem Zoom
inline constexpr int INITIAL_ITERATIONS = 100;     // Basiswert – empfohlen: 50 bis 200
inline constexpr int MAX_ITERATIONS_CAP = 50000;   // Hardlimit – je höher, desto langsamer, aber detaillierter
inline constexpr int ITERATION_STEP     = 5;       // Anstieg pro Zoomlevel – empfohlen: 1 bis 10

// 💚 CUDA Tile-Größen (neu quantisiert!)
inline constexpr int BASE_TILE_SIZE = 24; // Empfohlen: 16–32 – idealer Kompromiss aus Qualität & Performance
inline constexpr int MIN_TILE_SIZE  = 8;  // Untergrenze – kleinere Werte = feinere Analyse, aber höhere Last
inline constexpr int MAX_TILE_SIZE  = 64; // Obergrenze – größere Werte = weniger Rechenlast, aber ungenauer

// 🐅 Maus-Kommentar: Eigene clamp-Funktion, um <algorithm> Konflikte mit std::clamp zu umgehen.
// Eingesetzt zur Begrenzung dynamischer Parameter – robust auch ohne STL.
inline float my_clamp(float val, float minVal, float maxVal) {
    return (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
}

} // namespace Settings
