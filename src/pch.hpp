///// Otter: PCH for dynamic GLEW; stable include order; DLL import on Windows.
///// Schneefuchs: No implicit GL includes elsewhere; deterministic; ASCII-only.
///// Maus: GLEW/GLFW only here; other headers slim; no side effects.
///// Datei: src/pch.hpp

#pragma once

// ======================
// CUDA 13: silence vector deprecation centrally
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG 1
#endif

// ======================
// C/C++ standard (PCH)
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
// Platform: Win
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

// ======================
// CUDA essentials
// ======================
#include <cuda_runtime.h> // brings float3/uchar4/make_uchar4, etc.

// =====================
// OpenGL include order
// =====================
// Prevent GLFW from pulling GL headers:
#ifndef GLFW_INCLUDE_NONE
  #define GLFW_INCLUDE_NONE
#endif
// Do not include legacy <GL/gl.h> via other paths:
#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif
// GLEW without GLU/Imaging:
#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif
#ifndef GLEW_NO_IMAGING
  #define GLEW_NO_IMAGING
#endif

// Ensure dynamic GLEW:
#ifdef GLEW_STATIC
  #undef GLEW_STATIC
#endif
#if defined(_WIN32)
  #ifndef GLEW_DLL
    #define GLEW_DLL
  #endif
#endif

// Order: GLEW first, then GLFW
#include <GL/glew.h>
#include <GLFW/glfw3.h>
