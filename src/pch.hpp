///// Otter: PCH für dynamisches GLEW; stabile Include-Reihenfolge; DLL-Import unter Windows.
///// Schneefuchs: Keine impliziten GL-Includes; deterministisch; ASCII-only.
///// Maus: GLEW/GLFW nur hier; andere Header schlank halten; keine Seiteneffekte.

#pragma once

// ======================
// C/C++ Standard (PCH)
// ======================
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ===============
// Plattform: Win
// ===============
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

// =====================
// OpenGL include order
// =====================
// Verhindert, dass GLFW eigene GL-Header nachlädt:
#ifndef GLFW_INCLUDE_NONE
  #define GLFW_INCLUDE_NONE
#endif
// Keine Alt-GL-Header über GLEW einziehen:
#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif
// GLEW ohne GLU/Imaging (wird nicht benötigt):
#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif
#ifndef GLEW_NO_IMAGING
  #define GLEW_NO_IMAGING
#endif

// Erzwinge saubere DLL-Imports, wenn dynamisches GLEW genutzt wird (Windows).
#if defined(_WIN32) && !defined(GLEW_STATIC)
  #ifndef GLEW_DLL
    #define GLEW_DLL
  #endif
#endif

// Hinweis zur Link-Policy: Projekt bevorzugt dynamisches GLEW.
// Falls der Build dennoch -DGLEW_STATIC setzt, ist das hier nur eine Info.
#if defined(GLEW_STATIC)
  #pragma message("WARNING: GLEW_STATIC is defined by the build. Project policy prefers dynamic GLEW (DLL).")
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
