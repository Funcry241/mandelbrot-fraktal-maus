#pragma once

// settings.hpp — Alle zentralen Konstanten kompakt & verständlich kommentiert

#include <cmath>   // für logf, log2f, sqrtf
#include <cstdio>  // für printf bei Debug

namespace Settings {

inline constexpr bool debugGradient = false;
inline constexpr bool debugLogging  = true;

inline constexpr int width        = 1024;
inline constexpr int height       = 768;
inline constexpr int windowPosX   = 100;
inline constexpr int windowPosY   = 100;

inline constexpr float initialZoom    = 300.0f;
inline constexpr float zoomFactor     = 1.01f;
inline constexpr float initialOffsetX = -0.5f;
inline constexpr float initialOffsetY =  0.0f;

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;
inline constexpr float ZOOM_STEP_FACTOR   = 0.002f;

inline constexpr float MIN_OFFSET_STEP = 1e-8f;
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;

inline constexpr float VARIANCE_THRESHOLD      = 1e-12f;
inline constexpr float MIN_VARIANCE_THRESHOLD  = 1e-10f;

inline constexpr float AUTOZOOM_SPEED = 1.01f;

inline constexpr float DYNAMIC_RADIUS_SCALE = 1.0f;
inline constexpr int   DYNAMIC_RADIUS_MIN   = 20;
inline constexpr int   DYNAMIC_RADIUS_MAX   = 300;

inline constexpr int INITIAL_ITERATIONS = 100;
inline constexpr int MAX_ITERATIONS_CAP = 5000;
inline constexpr int ITERATION_STEP     = 5;

inline constexpr float LERP_FACTOR = 0.02f;

inline constexpr int BASE_TILE_SIZE = 8;
inline constexpr int MIN_TILE_SIZE  = 4;
inline constexpr int MAX_TILE_SIZE  = 32;

inline constexpr int TILE_W = 16;
inline constexpr int TILE_H = 16;

// 🧠 Eigene min()-Funktion zur Vermeidung von <algorithm>
inline int my_min(int a, int b) {
    return (a < b) ? a : b;
}

inline int dynamicTileSize(float zoom) {
    static int lastSize = -1;

    float logZoom = log10f(zoom + 1.0f);
    float rawSize = BASE_TILE_SIZE * (8.0f / (logZoom + 1.0f));

    constexpr int allowedSizes[] = {32, 16, 8, 4};

    int bestSize = allowedSizes[0];
    for (int size : allowedSizes) {
        if (rawSize >= size) {
            bestSize = size;
            break;
        }
    }

    if (bestSize != lastSize) {
#if defined(DEBUG) || defined(_DEBUG)
    if (Settings::debugLogging) {
        std::printf("[DEBUG] TileSize changed to %d\n", bestSize);
    }
#endif
        lastSize = bestSize;
    }

    return bestSize;
}

// eigene clamp()-Variante (float)
inline float my_clamp(float x, float minVal, float maxVal) {
    return (x < minVal) ? minVal : (x > maxVal) ? maxVal : x;
}

// eigene clamp()-Variante (int)
inline int my_clamp(int x, int minVal, int maxVal) {
    return (x < minVal) ? minVal : (x > maxVal) ? maxVal : x;
}

inline float dynamicVarianceThreshold(float zoom) {
    float scaled = VARIANCE_THRESHOLD * (1.0f + 0.02f * log2f(zoom + 1.0f));
    return my_clamp(scaled, VARIANCE_THRESHOLD, MIN_VARIANCE_THRESHOLD * 10.0f);
}

inline int dynamicSearchRadius(float zoom) {
    float radius = DYNAMIC_RADIUS_SCALE * sqrtf(zoom);
    return my_clamp(static_cast<int>(radius), DYNAMIC_RADIUS_MIN, DYNAMIC_RADIUS_MAX);
}

inline int dynamicIterationLimit(float zoom) {
    float boost = 1.0f + 0.001f * zoom;
    int iterations = static_cast<int>(INITIAL_ITERATIONS * boost);
    return my_min(iterations, MAX_ITERATIONS_CAP);
}

} // namespace Settings
