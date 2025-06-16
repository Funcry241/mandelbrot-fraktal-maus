#pragma once

// settings.hpp — Alle zentralen Konstanten kompakt & verständlich kommentiert

#include <algorithm>  // für std::max, std::clamp
#include <cmath>      // für logf, log2f, sqrtf

namespace Settings {

inline constexpr bool debugGradient = false;
inline constexpr bool debugLogging  = true;

inline constexpr int width        = 1024;
inline constexpr int height       = 768;
inline constexpr int windowPosX   = 100;
inline constexpr int windowPosY   = 100;

inline constexpr float initialZoom    = 300.0f;
inline constexpr float initialOffsetX = -0.5f;
inline constexpr float initialOffsetY =  0.0f;

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;
inline constexpr float ZOOM_STEP_FACTOR   = 0.002f;

inline constexpr float MIN_OFFSET_STEP = 1e-8f;
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;

inline constexpr float VARIANCE_THRESHOLD      = 1e-12f;
inline constexpr float MIN_VARIANCE_THRESHOLD  = 1e-10f;

inline constexpr float AUTOZOOM_SPEED = 1.01f;

inline constexpr int INITIAL_ITERATIONS = 100;
inline constexpr int MAX_ITERATIONS_CAP = 5000;
inline constexpr int ITERATION_STEP     = 5;

inline constexpr float LERP_FACTOR = 0.02f;

inline constexpr int BASE_TILE_SIZE = 8;
inline constexpr int MIN_TILE_SIZE  = 4;
inline constexpr int MAX_TILE_SIZE  = 32;

inline constexpr int TILE_W = 16;
inline constexpr int TILE_H = 16;

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

inline float dynamicVarianceThreshold(float zoom) {
    float scaled = VARIANCE_THRESHOLD * (1.0f + 0.02f * log2f(zoom + 1.0f));
    return std::clamp(scaled, VARIANCE_THRESHOLD, MIN_VARIANCE_THRESHOLD * 10.0f);
}

} // namespace Settings
