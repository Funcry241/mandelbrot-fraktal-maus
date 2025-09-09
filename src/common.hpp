///// Otter: Zentrale Helfer & Konstanten; praezise Tilegroesse via log1p, keine eigenen CHECK-Makros.
///// Schneefuchs: Deterministisch, ASCII-only; Header/Source synchron; keine verdeckten Funktionswechsel.
///// Maus: Nur LUCHS_LOG_* fuers Logging; Settings steuern Verhalten; keine Supersampling-Pfade.
///// Datei: src/common.hpp

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

#ifndef GL_DO_NOT_INCLUDE_GL_H
  #define GL_DO_NOT_INCLUDE_GL_H
#endif
#ifndef GLEW_NO_GLU
  #define GLEW_NO_GLU
#endif

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <ctime>

#include "luchs_log_host.hpp"   // <- stellt CUDA_CHECK bereit
#include "settings.hpp"

// --- KEIN eigenes CUDA_CHECK mehr hier ---

// 🦊 Schneefuchs: numerisch stabiler mit log1p fuer kleine zoom-Werte; deterministisch geklemmt.
// 🦦 Otter: INV_LN2 als constexpr vermeidet wiederholte std::log(2.0)-Aufrufe.
[[nodiscard]] inline int computeTileSizeFromZoom(float zoom) noexcept {
    const float base = static_cast<float>(Settings::BASE_TILE_SIZE);
    const float z    = std::max(0.0f, zoom);
    constexpr float INV_LN2 = 1.4426950408889634f; // 1 / ln(2)
    const float raw  = base - std::log1p(z) * INV_LN2; // ≈ base - log2(1+z)
    int t = static_cast<int>(std::lround(raw));
    t = std::clamp(t, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);
    return t;
}
