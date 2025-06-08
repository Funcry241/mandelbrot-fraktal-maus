#pragma once

// settings.hpp — 🐭 Alle zentralen Konstanten kompakt & verständlich gesammelt

namespace Settings {

// 🛠️ Debugging / Test-Modus
inline constexpr bool debugGradient = false;   
// Aktiviert ein einfaches Farb-Gradient-Testbild statt Mandelbrot (zur Fehleranalyse)

inline constexpr bool debugLogging  = true;    
// Schreibt detaillierte Debug-Informationen während des Renderings in die Konsole

// 🖥️ Fenster und Bild
inline constexpr int width        = 1024;       
// Breite des Fensters in Pixeln

inline constexpr int height       = 768;        
// Höhe des Fensters in Pixeln

inline constexpr int windowPosX   = 100;        
inline constexpr int windowPosY   = 100;        
// Fenster-Startposition auf dem Bildschirm

// 🔎 Zoom & Pan Einstellungen
inline constexpr float initialZoom    = 300.0f;  
// Start-Zoomstufe — höherer Wert bedeutet stärkerer initialer Zoom ins Fraktal

inline constexpr float zoomFactor     = 1.01f;    
// Multiplikator für Zoom-Increment pro Frame (bei manuellem Zoom)

inline constexpr float initialOffsetX = -0.5f;    
inline constexpr float initialOffsetY =  0.0f;    
// Start-Offset im Fraktal — steuert den initialen Bildausschnitt

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;     
// Schrittweite für Offset-Verschiebungen beim Pan (Tastatursteuerung)

inline constexpr float ZOOM_STEP_FACTOR   = 0.002f;    
// Prozentuale Erhöhung des Zooms pro Frame bei Auto-Zoom

inline constexpr float MIN_OFFSET_STEP = 1e-8f;       
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;       
// Untergrenzen für Offset- und Zoom-Änderungen (um "zitternde" Bewegungen zu verhindern)

// 🧠 Auto-Zoom Steuerung
inline constexpr float VARIANCE_THRESHOLD = 1e-12f;   
// Basis-Schwelle für die Komplexität eines Bildausschnitts — niedrige Werte sind empfindlicher

// Dynamischer Variance-Threshold in Abhängigkeit vom Zoom
inline float dynamicVarianceThreshold(float zoom) {
    // Passt die Schwelle logarithmisch an den Zoom an: je höher der Zoom, desto kleiner der Schwellenwert
    return VARIANCE_THRESHOLD / logf(zoom + 2.0f);
}

// 🔢 Iterations-Steuerung
inline constexpr int TILE_W             = 8;    
inline constexpr int TILE_H             = 8;    
// Breite und Höhe einer Kachel (Tile) zur lokalen Variabilitätsanalyse

inline constexpr int INITIAL_ITERATIONS = 100;  
// Startanzahl der Iterationen für die Mandelbrot-Berechnung

inline constexpr int MAX_ITERATIONS_CAP = 5000; 
// Obergrenze für Iterationen — schützt vor extrem langen Berechnungen

inline constexpr int ITERATION_STEP     = 5;    
// Schrittweite, mit der die Iterationsanzahl erhöht wird, wenn sich der Zoom verstärkt

// 🐭 Sanftes Gliding für Offset-Änderungen
inline constexpr float LERP_FACTOR      = 0.02f;  
// Interpolationsfaktor für weiches Nachführen des Offsets (für sanfte Bildbewegungen)

// 📈 Dynamischer Suchradius für Auto-Zoom
inline constexpr float DYNAMIC_RADIUS_SCALE = 1.5f;   
// Skalierungsfaktor für den Suchradius basierend auf sqrt(Zoom) — höhere Werte durchsuchen ein größeres Gebiet

inline constexpr int   DYNAMIC_RADIUS_MIN   = 30;     
// Minimaler Radius für die Suche nach komplexen Bildbereichen

inline constexpr int   DYNAMIC_RADIUS_MAX   = 2000;   
// Maximaler Radius für die Suche — begrenzt die Rechenzeit und verhindert "Ausfransen"

} // namespace Settings
