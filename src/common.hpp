// Datei: src/common.hpp
#pragma once

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif
#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <ctime>

#include "luchs_log_host.hpp"   // <- stellt CUDA_CHECK bereit
#include "settings.hpp"

// --- KEIN eigenes CUDA_CHECK mehr hier ---

// 🦊 Schneefuchs: Nutze std::log2 (C++-konform) statt std::log2f; gleiche Semantik, bessere Portabilität.
// 🦦 Otter: Pfad bleibt numerisch stabil; Clamp nach Rundung sichert Grenzen.
inline int computeTileSizeFromZoom(float zoom) noexcept {
    const float base = static_cast<float>(Settings::BASE_TILE_SIZE);
    const float raw  = base - std::log2(std::max(0.0f, zoom) + 1.0f);
    int t = static_cast<int>(std::lround(raw));
    t = std::clamp(t, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);
    return t;
}

inline bool getLocalTime(std::tm& outTm, std::time_t t) noexcept {
#if defined(_WIN32)
    return localtime_s(&outTm, &t) == 0;
#else
    return localtime_r(&t, &outTm) != nullptr;
#endif
}
