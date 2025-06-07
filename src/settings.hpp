#pragma once

#include <cmath>

namespace Settings {

// Debug
inline constexpr bool debugGradient = false, debugLogging = false;

// Fenster
inline constexpr int width = 1024, height = 768, windowPosX = 100, windowPosY = 100;

// Zoom & Pan
inline constexpr float initialZoom = 300.0f, zoomFactor = 1.01f;
inline constexpr float initialOffsetX = -0.5f, initialOffsetY = 0.0f;
inline constexpr float OFFSET_STEP_FACTOR = 0.5f, ZOOM_STEP_FACTOR = 0.15f;
inline constexpr float MIN_OFFSET_STEP = 1e-8f, MIN_ZOOM_STEP = 1e-6f;

// Auto-Zoom
inline constexpr float VARIANCE_THRESHOLD = 1e-12f;
inline float dynamicVarianceThreshold(float zoom) {   // ✅ KEIN constexpr
    return VARIANCE_THRESHOLD / std::log(zoom + 2.0f);
}

// Iterationen
inline constexpr int TILE_W = 16, TILE_H = 16;
inline constexpr int INITIAL_ITERATIONS = 100, MAX_ITERATIONS_CAP = 5000, ITERATION_STEP = 5;

} // namespace Settings
