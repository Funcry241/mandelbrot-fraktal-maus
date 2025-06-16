// Datei: src/common.hpp
// 🐭 Maus-Kommentar: Zentrale Header-Schutzmauer für CUDA, OpenGL, Windows, C++

#pragma once

// 🔧 Windows-Makros beschneiden
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

// 🔧 GLEW ohne gl.h, kein GLU
#define GL_DO_NOT_INCLUDE_GL_H
#define GLEW_NO_GLU

// 🪟 Windows-API
#include <windows.h>

// 🎨 OpenGL
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
