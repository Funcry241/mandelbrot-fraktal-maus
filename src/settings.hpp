#pragma once

// ------------------------------------------------------------------------------------------------
// settings.hpp
// Definiert alle statischen Konfigurationswerte für Fenstergröße, Zoom-Verhalten, Pan, Tile-Größe usw.
// ------------------------------------------------------------------------------------------------

namespace Settings {

    // Debug-Modus (für Debug-Gradient)
    inline constexpr bool debugGradient = false;

    // 🐭 NEU: Debug-Logging (für CUDA Logging, optional)
    inline constexpr bool debugLogging = false; // true = viel Konsolenausgabe, false = Ruhe

    // Fenstergröße
    inline constexpr int width  = 1024; // Fensterbreite
    inline constexpr int height = 768;  // Fensterhöhe

    // Zoom-Parameter
    inline constexpr float initialZoom = 300.0f;   // Start-Zoom
    inline constexpr float zoomFactor  = 1.01f;    // Zoom-Multiplikator pro Frame
    inline constexpr float minScale    = 1e-20f;   // Minimale Skalierung (keine NaNs)

    // Pan-Parameter
    inline constexpr float panFraction = 0.1f;     // Anteil des Zoom-Bereichs für Pan

    // Mandelbrot-Parameter
    inline constexpr int maxIterations = 500;      // Maximale Iterationszahl pro Pixel

    // Tile-Größe für Complexity-Kernel (muss in allen Dateien gleich sein)
    inline constexpr int TILE_W = 16;
    inline constexpr int TILE_H = 16;

    // (Optional) Fensterposition (z.B. für Multi-Monitor-Setup)
    inline constexpr int windowPosX = 100;
    inline constexpr int windowPosY = 100;

    // Schwellwert für dynamische Verfeinerung (ob ein Tile weiterverfeinert wird)
    inline constexpr float DYNAMIC_THRESHOLD = 400.0f;

    // 🐭 Offset-Startposition
    inline constexpr float initialOffsetX = -0.5f;
    inline constexpr float initialOffsetY =  0.0f;

    // 🐭 Schwenk- und Zoom-Parameter
    inline constexpr float OFFSET_STEP_FACTOR = 0.50f;  // Basis-Schrittweite für Offset
    inline constexpr float ZOOM_STEP_FACTOR   = 0.15f;  // Basis-Schrittweite für Zoom

    // 🐭 NEU: Minimalgrößen für sanftes Verhalten bei sehr großem Zoom
    inline constexpr float MIN_OFFSET_STEP = 1e-8f;     // Kleinster erlaubter Offset-Schritt
    inline constexpr float MIN_ZOOM_STEP   = 1e-6f;     // Kleinster erlaubter Zoom-Schritt

    // 🐭 NEU: Varianz-Schwelle für Tile-Selektion
    inline constexpr float VARIANCE_THRESHOLD = 1e-12f; // Früher 1e-6f → jetzt empfindlicher!
}
