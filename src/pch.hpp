// Datei: src/pch.hpp
// 🐭 Maus-Kommentar: Precompiled Header – Projekt Phönix! FreeType raus, EasyFont rein. OpenGL bleibt dominant. CUDA-Interop bleibt minimalinvasiv. Schneefuchs sieht, Otter lacht.

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

// OpenGL: GLEW IMMER vor GLFW einbinden!
#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA: nur Runtime/Interop – kein cuda.h, kein driver_api!
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Standardbibliothek
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm> // optional für std::max etc.

// 🧠 Projektmodule für IntelliSense-Boost (PCH-safe)
#include "settings.hpp"
#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "frame_pipeline.hpp"
#include "heatmap_overlay.hpp"

// Leerzeile am Ende zum sauberen Abschluss
