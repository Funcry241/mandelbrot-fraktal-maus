///// Otter: Nacktmull-Shade (Header) - inline Device-Helfer + Kernel-Prototypen, keine Implementierungen.
///// Schneefuchs: Deterministisch, ASCII-only; nur CUDA-Basics; Header/Source strikt getrennt.
///// Maus: Keine eigenen Typ-Redefs (NV-Types verwenden); API stabil; mikro-optimierte Inlines.

#pragma once

// CUDA / Vektor-Typen
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h> // make_float3, make_uchar4

// Standard-Math (statt internal math_functions.h)
#include <cmath>

// ------------------------ kleine Device-Helfer ---------------------------------
#if defined(__CUDACC__)
static __device__ __forceinline__ float clamp01(float x) {
    // Bewusst simpel & stabil:
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

static __device__ __forceinline__ uchar4 make_rgba_01(float r, float g, float b, float a = 1.f) {
    r = clamp01(r); g = clamp01(g); b = clamp01(b); a = clamp01(a);
    return make_uchar4(
        (unsigned char)(r * 255.f + 0.5f),
        (unsigned char)(g * 255.f + 0.5f),
        (unsigned char)(b * 255.f + 0.5f),
        (unsigned char)(a * 255.f + 0.5f)
    );
}

static __device__ __forceinline__ float3 hsv2rgb(float h, float s, float v) {
    // h in [0,1), s,v in [0,1]
    float r = v, g = v, b = v;
    if (s > 0.f) {
        h = (h - ::floorf(h)) * 6.f; // [0,6)
        const int   i = (int)h;
        const float f = h - (float)i;
        const float p = v * (1.f - s);
        const float q = v * (1.f - s * f);
        const float t = v * (1.f - s * (1.f - f));
        switch (i) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break; // i==5
        }
    }
    return make_float3(r, g, b);
}
#endif // __CUDACC__

// ------------------------ Kernel-Prototypen ------------------------------------
#if defined(__CUDACC__)
extern "C" __global__
void shade_from_iterations(uchar4* surface,
                           const int* __restrict__ iters,
                           int width, int height,
                           int maxIterations);

extern "C" __global__
void shade_test_pattern(uchar4* surface,
                        int width, int height,
                        int checkSize);
#endif // __CUDACC__
