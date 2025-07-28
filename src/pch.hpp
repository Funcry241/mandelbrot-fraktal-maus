// Datei: src/pch.hpp
// 🐭 Maus-Kommentar: Reine Host-PCH. Otter: Kompiliert in jeder .cpp-Datei. Schneefuchs: Keine CUDA-Fallen.

#pragma once

// C/C++ Standard
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cstring>

// Plattform
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#endif

// OpenGL (für GUI/HUD/UI etc.)
#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>
