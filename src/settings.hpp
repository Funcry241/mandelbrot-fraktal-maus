#pragma once

// ------------------------------------------------------------------------------------------------
// settings.hpp
// Definiert alle statischen Konfigurationswerte für Fenstergröße, Zoom-Verhalten, Pan, Tile-Größe usw.
// ------------------------------------------------------------------------------------------------

namespace Settings {

    // Debug-Modus (für Debug-Gradient)
    inline constexpr bool debugGradient = false; 

    // Fenstergröße
    inline constexpr int width  = 1024;  // Fensterbreite
    inline constexpr int height =  768;  // Fensterhöhe

    // Zoom-Parameter
    inline constexpr float initialZoom = 300.0f;  // Start-Zoom
    inline constexpr float zoomFactor  =   1.01f; // Zoom-Multiplikator pro Frame
    inline constexpr float minScale    = 1e-20f;  // Minimale Skalierung zum Verhindern von NaNs (keine 0-Division)

    // Pan-Parameter
    inline constexpr float panFraction = 0.1f;    // Anteil des Zoom-Bereichs, um den bei Pan verschoben wird

    // Mandelbrot-Parameter
    inline constexpr int maxIterations = 500;     // Maximale Iterationszahl pro Pixel

    // Tile-Größe für Complexity-Kernel (muss in allen Dateien gleich sein)
    inline constexpr int TILE_W = 16;
    inline constexpr int TILE_H = 16;

    // (Optional) Fensterposition (z.B. für Multi-Monitor-Setup)
    inline constexpr int windowPosX = 100;        // Fenster Start-X (optional)
    inline constexpr int windowPosY = 100;        // Fenster Start-Y (optional)

    // Schwellwert für dynamische Verfeinerung (NEU)
    inline constexpr float DYNAMIC_THRESHOLD = 400.0f;

    // 🐭 NEU: Initiale Offset-Position – klassisch bei Mandelbrot
    inline constexpr float initialOffsetX = -0.5f;
    inline constexpr float initialOffsetY = 0.0f;
}
