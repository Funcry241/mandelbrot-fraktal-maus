// Datei: src/common.hpp
// 🐭 Maus-Kommentar: Zentraler Header mit weicher Tile-Größenfunktion, CUDA-Fehlermakro und Standard-Includes.
// Schneefuchs empfiehlt: kein doppeltes <cmath>, computeTileSizeFromZoom immer aus genau dieser Quelle verwenden! Alles warnfrei, keine Doppel-Defines, keine Header-Schatten.

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

// 🦾 Logging
#include "luchs_log_host.hpp"

// 🧪 CUDA-Fehlerprüfung
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            LUCHS_LOG_HOST("[CUDA ERROR] %s", cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 🔢 Tile-Größe aus Zoom-Level berechnen (weicher Verlauf)
inline int computeTileSizeFromZoom(float zoom) {
    float raw = 32.0f - std::log2f(zoom + 1.0f);  // weich fallend
    int clamped = std::max(4, std::min(256, static_cast<int>(std::round(raw))));
    return clamped;
}
