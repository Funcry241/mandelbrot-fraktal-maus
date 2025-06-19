// Datei: src/common.hpp
// Zeilen: +15
// 🐭 Maus-Kommentar: Tile-Größe wird jetzt logarithmisch aus dem Zoomfaktor berechnet – ohne Sprungstellen, kontinuierlich gleitend. Schneefuchs flüstert: „Wer weich zoomt, gewinnt mehr Spielraum.“

#pragma once
#include <cmath>

inline int computeTileSizeFromZoom(float zoom) {
    float raw = 32.0f - std::log2f(zoom + 1.0f);  // weich fallend
    int clamped = std::max(4, std::min(32, static_cast<int>(std::round(raw))));
    return clamped;
}

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

// 🎨 OpenGL: GLEW vor gl.h, kein GLU
#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif

#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif

#include <GL/glew.h>

// ⚡ CUDA
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

// 🧪 CUDA-Fehlerprüfung
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
