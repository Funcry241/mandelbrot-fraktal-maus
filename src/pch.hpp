// 🐭 Maus-Kommentar: Precompiled Header für Windows + OpenGL + CUDA – sorgt für stabile Symboldefinitionen & Reihenfolge

#pragma once

// Wichtig: Windows-API zuerst, minimal halten
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

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
