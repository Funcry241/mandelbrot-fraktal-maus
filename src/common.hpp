// Datei: src/common.hpp
// 🐭 Maus-Kommentar: Jetzt mit deterministischem CUDA_CHECK. Keine stderr-Leichen mehr. Nur ASCII. Nur Klartext.
// 🦦 Otter: Keine stille Panik mehr. Jeder Fehler hat einen Pfad.
// 🦊 Schneefuchs: Sichtbarkeit vor Geschwindigkeit. Logging ist Debugging.
#pragma once

// 🔧 Windows-spezifische Makros und Header
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

// 🎨 OpenGL: GLEW vor glfw3.h, kein GLU
#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif

#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif

#include <cuda_gl_interop.h>

// 🧠 C++ Standardbibliothek
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <ctime>

// 🦾 Logging
#include "luchs_log_host.hpp"

// 🧪 CUDA-Fehlerprüfung – deterministisch, sichtbar, ASCII-only
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            LUCHS_LOG_HOST("[CUDA ERROR] %s failed at %s:%d → %s",             \
                           #expr, __FILE__, __LINE__, cudaGetErrorString(err__)); \
            throw std::runtime_error("CUDA failure: " #expr);                  \
        }                                                                      \
    } while (0)

// 🔢 Tile-Größe aus Zoom-Level berechnen (weicher Verlauf)
inline int computeTileSizeFromZoom(float zoom) {
    float raw = 32.0f - std::log2f(zoom + 1.0f);  // weich fallend
    int clamped = std::max(4, std::min(256, static_cast<int>(std::round(raw))));
    return clamped;
}

// ⏰ Plattformübergreifend threadsicheres localtime
inline bool getLocalTime(std::tm& outTm, std::time_t t) {
#if defined(_WIN32)
    return localtime_s(&outTm, &t) == 0;
#else
    return localtime_r(&t, &outTm) != nullptr;
#endif
}
