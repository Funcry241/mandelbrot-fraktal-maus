// 🐭 Maus-Kommentar: Precompiled Header für Windows + OpenGL + CUDA – sorgt für stabile Symboldefinitionen & Reihenfolge

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

// GLEW vor GLFW, um GL.h-Header-Konflikte zu vermeiden
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA Runtime – aber **nicht** driver_api.h
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// C++ Standardbibliothek
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
