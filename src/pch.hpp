// Datei: src/pch.hpp
// Zeilen: 32
// 🐭 Maus-Kommentar: Precompiled Header – klar strukturiert: Windows-Header mit WIN32-Defines, dann OpenGL (GLEW+GLFW), dann CUDA-Runtime, dann STL. Achtung: `windows.h` nur mit `NOMINMAX`, `GLEW` nur vor `GLFW`. Schneefuchs bestand darauf, dass CUDA *nur* Runtime einbindet – nie `cuda.h` – sonst PCH-Krater.

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

// OpenGL: Reihenfolge essenziell
#include <GL/glew.h>         // Muss **vor** glfw3.h kommen
#include <GLFW/glfw3.h>

// CUDA: nur Runtime/Interop – kein cuda.h, kein driver_api
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Standardbibliothek
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
