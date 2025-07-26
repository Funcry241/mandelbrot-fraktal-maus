// Datei: src/settings.hpp
// 🐅 Maus-Kommentar: Steuerungszentrale für Auto-Zoom, Fraktal-Feintuning, Entropieanalyse und CUDA-Tile-Verhalten.
// Nur aktive, genutzte Konstanten bleiben erhalten - der Rest wurde entrümpelt. Die Schwester atmet auf.
// Modernisiert mit robusten Kommentaren und eigenem clamp, [[nodiscard]] wurde an Variablen entfernt (nvcc inkompatibel).

#pragma once

#include <cmath> // für logf, log2f, sqrtf

namespace Settings {

// 🔍 Zoom-Faktor bei jedem Auto-Zoom-Schritt.
// Empfohlen: 1.05 (langsam), 1.1 (moderat), 1.2+ (aggressiv).
// Höhere Werte vergrößern den Bildausschnitt schneller, aber riskieren visuelle Artefakte.
constexpr float zoomFactor = 1.07f;

// Untere Entropie-Schwelle für Auto-Zoom-Zielauswahl.
// Nur Tiles mit Entropie > ENTROPY_THRESHOLD_LOW werden als Kandidaten betrachtet.
//
// Wertempfehlung:
//   - 0.0f: kein Filter (auch langweilige Bereiche werden berücksichtigt)
//   - 2.0f-3.0f: realistische Schwelle für kontrastarme Tiles
//   - 4.0f-5.0f: nur sehr strukturierte Bereiche
//
// Erhöhung → stärkerer Filter, langsameres Zoomen  
// Reduktion → breitere Auswahl, aber potenziell uninteressanter Zoom
inline constexpr float ENTROPY_THRESHOLD_LOW = 2.5f;

// 🔍 Debug-Modi: visuelle Darstellung & Konsolen-Ausgabe aktivieren
constexpr bool debugGradient = false; // Aktiviert reine Entropie-Ansicht (keine Farben) - nur zu Analysezwecken
constexpr bool debugLogging  = true;  // Aktiviert Konsolenausgaben für Auto-Zoom, Tile-Entropie etc.

// 🔥 Sichtbarkeit des Heatmap-Overlays beim Programmstart
// true = Heatmap (Entropie-Kontrast) ist sofort sichtbar
// false = Muss per Taste H aktiviert werden
constexpr bool heatmapOverlayEnabled = true; // Otter: standardmäßig an

// 🐷 Sichtbarkeit des WarzenschweinOverlays (Text-HUD) beim Programmstart
// true  = HUD mit FPS, Zoom etc. ist sofort sichtbar
// false = Muss per Taste aktiviert oder im Code gesetzt werden
constexpr bool warzenschweinOverlayEnabled = true; // Otter: HUD ab Start sichtbar

// HUD-Textgröße (in NDC-Einheiten pro Pixelquadrat)
// Empfohlen: 0.0015 (klein), 0.0025 (normal), 0.004 (groß)
// Wirkt sich auf WarzenschweinOverlay aus
inline constexpr float hudPixelSize = 0.0025f;

// 💥 Fensterkonfiguration (Initialgröße und Position auf dem Bildschirm)
constexpr int width       = 1024; // Breite des Fensters in Pixel - empfohlen: 800 bis 1600
constexpr int height      = 768;  // Höhe des Fensters in Pixel - empfohlen: 600 bis 1200
constexpr int windowPosX  = 100;  // Startposition links
constexpr int windowPosY  = 100;  // Startposition oben

// 🔭 Initialer Fraktal-Ausschnitt (Zoomfaktor und Verschiebung)
constexpr float initialZoom    = 1.5f;  // Start-Zoom: höherer Wert = näher dran - empfohlen: 100-1000
constexpr float initialOffsetX = -0.5f; // Anfangsverschiebung auf der X-Achse
constexpr float initialOffsetY = 0.0f;  // Anfangsverschiebung auf der Y-Achse

// 🔍 Manueller Zoom (per Mausrad oder Tasten) - pro Schritt
constexpr float ZOOM_STEP_FACTOR = 0.002f; // Erhöhung = schnelleres Zoomen - empfohlen: 0.001 bis 0.01

// 🌟 Schwellenwerte für Entropieanalyse zur Auswahl interessanter Tiles
constexpr float VARIANCE_THRESHOLD     = 0.01f;  // Hauptschwelle für interessante Tiles - je kleiner, desto empfindlicher
constexpr float MIN_VARIANCE_THRESHOLD = 0.001f; // Notbremse für zu starkes Auto-Zoom - empfohlen: 1e-10 bis 1e-8

// 🌀 Geschwindigkeit des automatischen Zooms pro Frame
constexpr float AUTOZOOM_SPEED = 1.005f; // Höher = schnellerer Zoom - empfohlen: 1.002 bis 1.01

// Minimaler Abstand (in Fraktalkoordinaten) für Zielwechsel bei Auto-Zoom
// Empfehlung: 0.0001 bis 0.01 je nach Zoomstufe - kleiner = empfindlicher, größer = träger
constexpr float MIN_JUMP_DISTANCE = 0.001f;

// 🪎 Glättungsfaktor für Kamera-Nachführung zum Ziel (linearer LERP)
// Kleiner = langsameres Nachziehen, größer = schneller & unruhiger
constexpr float LERP_FACTOR = 0.035f; // empfohlen: 0.01 bis 0.08

// 🦕 Stillstandsschwelle für Offset-Bewegung - wirkt wie ein Ruhepuffer
// Wenn Offset näher als DEADZONE am Ziel liegt, wird keine Bewegung mehr ausgeführt
constexpr float DEADZONE = 1e-8f; // empfohlen: 1e-10 bis 1e-8 - kleiner = empfindlicher

// 🦕 Maximaler Anteil der Ziel-Distanz, der pro Frame bewegt werden darf (in Fraktal-Koordinaten)
// Limitiert Bewegungsgeschwindigkeit zusätzlich zur tanh-Dämpfung
constexpr float MAX_OFFSET_FRACTION = 0.1f; // empfohlen: 0.05 bis 0.2 - größer = schnelleres Nachziehen

// 📈 Bonusgewichtung für Tiles, die nah am aktuellen Offset liegen (für stabileres Auto-Zoom)
// 0.0 = keine Bevorzugung, 1.0 = starker Bias auf Nähe
constexpr float ENTROPY_NEARBY_BIAS = 0.3f; // empfohlen: 0.0 bis 0.6 - höher = weniger Hüpfen

// 🦕 Skaliert die Offset-Distanz vor Anwendung von tanh (nonlineare Dämpfung)
// Kleinere Werte = stärkere Dämpfung bei kleinen Bewegungen
// Empfohlen: 1.0 bis 10.0 - z.‍B. 5.0 bedeutet, dass bei tanh(5.0 * distance) ≈ 1 schnelle Bewegung erfolgt
constexpr float OFFSET_TANH_SCALE = 5.0f;

// ♻️ Iterationssteuerung - beeinflusst Detailtiefe bei starkem Zoom
constexpr int INITIAL_ITERATIONS = 100;    // Basiswert - empfohlen: 50 bis 200
constexpr int MAX_ITERATIONS_CAP = 50000;  // Hardlimit - je höher, desto langsamer, aber detaillierter

// 💚 CUDA Tile-Größen (neu quantisiert!)
constexpr int BASE_TILE_SIZE = 24; // Empfohlen: 16-32 - idealer Kompromiss aus Qualität & Performance
constexpr int MIN_TILE_SIZE  = 8;  // Untergrenze - kleinere Werte = feinere Analyse, aber höhere Last
constexpr int MAX_TILE_SIZE  = 64; // Obergrenze - größere Werte = weniger Rechenlast, aber ungenauer

// 🕊️ Adaptive LERP-Geschwindigkeit zwischen Kamera-Offset und Ziel
constexpr float ALPHA_LERP_MIN = 0.01f; // Kolibri
constexpr float ALPHA_LERP_MAX = 0.10f; // Kolibri

} // namespace Settings
