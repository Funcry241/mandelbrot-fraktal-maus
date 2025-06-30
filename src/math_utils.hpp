// Datei: src/math_utils.hpp
// Zeilen: 24
/* 🐭 interner Maus-Kommentar:
   Diese Version vermeidet Duplikate – make_float2 & make_double2 kommen von CUDA.
   Wir definieren nur zusätzliche Hilfsfunktionen wie distance, length etc.
*/

#pragma once
#include <cuda_runtime.h> // bringt float2, double2 + make_* mit

inline float length(const float2& f) {
    return std::sqrt(f.x * f.x + f.y * f.y);
}

inline float distance(const float2& a, const float2& b) {
    return length(make_float2(b.x - a.x, b.y - a.y));
}

// ggf. später: eigene lerp(), clamp(), normalize() etc.
