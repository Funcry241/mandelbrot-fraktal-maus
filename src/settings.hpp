#pragma once

// settings.hpp — Alle zentralen Konstanten kompakt & modern
namespace Settings {

inline constexpr bool debugGradient = false;
inline constexpr bool debugLogging  = false; // Viel Konsolenausgabe (optional)

inline constexpr int width  = 1024;
inline constexpr int height = 768;

inline constexpr float initialZoom = 300.0f;
inline constexpr float zoomFactor  = 1.01f;

inline constexpr int maxIterations = 500;
inline constexpr int TILE_W = 16;
inline constexpr int TILE_H = 16;

inline constexpr int windowPosX = 100;
inline constexpr int windowPosY = 100;

inline constexpr float initialOffsetX = -0.5f;
inline constexpr float initialOffsetY =  0.0f;

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;   // Basis-Offset pro Frame (wird mit 1/Zoom skaliert)
inline constexpr float ZOOM_STEP_FACTOR   = 0.15f;  // Basis-Zoomrate pro Frame

inline constexpr float MIN_OFFSET_STEP = 1e-8f;     // Minimal erlaubter Pan-Schritt
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;     // Minimal erlaubter Zoom-Schritt

inline constexpr float VARIANCE_THRESHOLD = 1e-12f; // Tile-Varianzschwelle für Auto-Zoom
}
