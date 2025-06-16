// Datei: src/common.hpp
// Zeilen: 50
// 🐭 Maus-Kommentar: Zentrale Header-Schutzmauer für CUDA, OpenGL, Windows und C++-Standard. Enthält essentielle Makros, pragmatische Includes und die `CUDA_CHECK`-Macro für robuste Fehlerbehandlung. Schneefuchs hätte darauf bestanden, dass kein `GLU` reinkriecht und der Fehler sauber mit Datei+Zeile rauskommt.

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

// 🎨 OpenGL: GLEW vor gl.h, kein GLU
#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif

#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif

#include <GL/glew.h>

// ⚡ CUDA
#include <cuda_runtime.h>
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
