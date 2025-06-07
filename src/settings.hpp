#pragma once

// ------------------------------------------------------------------------------------------------
// settings.hpp
// Definiert alle statischen Konfigurationswerte für Fenstergröße, Zoom-Verhalten, Pan, Tile-Größe usw.
// ------------------------------------------------------------------------------------------------

namespace Settings {

    // Debug-Modus (für Debug-Gradient)
    inline constexpr bool debugGradient = false;

    // 🐭 Debug-Logging (für CUDA-Logausgabe, optional)
    inline constexpr bool debugLogging = false; // true = viel Konsolenausgabe, false = ruhig

    // Fenstergröße
    inline constexpr int width  = 1024; // Fensterbreite
    inline constexpr int height = 768;  // Fensterhöhe

    // Zoom-Parameter
    inline constexpr float initialZoom = 300.0f;   // Start-Zoom
    inline constexpr float zoomFactor  = 1.01f;    // Zoom-Multiplikator pro Frame
    inline constexpr float minScale    = 1e-20f;   // Minimale Skalierung (gegen NaNs absichern)

    // Pan-Parameter
    inline constexpr float panFraction = 0.1f;     // Anteil des Zoom-Bereichs für Pan

    // Mandelbrot-Parameter
    inline constexpr int maxIterations = 500;      // Maximale Iterationszahl pro Pixel

    // Tile-Größe für Complexity-Kernel (muss in allen Dateien gleich sein)
    inline constexpr int TILE_W = 16;
    inline constexpr int TILE_H = 16;

    // Fenster-Startposition (optional, für Multi-Monitor-Setups)
    inline constexpr int windowPosX = 100;
    inline constexpr int windowPosY = 100;

    // Schwellwert für dynamische Verfeinerung
    inline constexpr float DYNAMIC_THRESHOLD = 400.0f; // Durchschnittliche Iterationen pro Tile

    // 🐭 Offset-Startposition (zentriert aufs typische Mandelbrot-Zentrum)
    inline constexpr float initialOffsetX = -0.5f;
    inline constexpr float initialOffsetY =  0.0f;

    // 🐭 Schwenk- und Zoom-Parameter (dynamisch abhängig von Zoomstufe)
    inline constexpr float OFFSET_STEP_FACTOR = 0.50f;  // Schrittweite für Offset pro Frame
    inline constexpr float ZOOM_STEP_FACTOR   = 0.15f;  // Schrittweite für Zoom pro Frame

    // 🐭 Minimalwerte für Bewegung/Zoom – verhindern "Einfrieren" bei extremem Zoom
    inline constexpr float MIN_OFFSET_STEP = 1e-8f;     // Kleinster erlaubter Offset-Schritt
    inline constexpr float MIN_ZOOM_STEP   = 1e-6f;     // Kleinster erlaubter Zoom-Schritt

    // 🐭 Varianzschwelle für die Tile-Selektion (wie empfindlich "Interessantes" erkannt wird)
    inline constexpr float VARIANCE_THRESHOLD = 1e-12f; // (je kleiner, desto empfindlicher)
}
