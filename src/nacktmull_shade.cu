// Datei: src/nacktmull_shade.cu
// MAUS: Implementation for Nacktmull shading kernels (no uchar4 redefinition)

#include <cuda_runtime.h>
#include <vector_types.h>
#include "nacktmull_shade.cuh"

// 🦦 Otter: keine eigene uchar4-Struct – wir verwenden die NV-Types.
// 🦊 Schneefuchs: simples, deterministisches Farbschema; keine Device-Logs.

static __device__ __forceinline__ uchar4 pack_rgba(
    unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255u
) {
    uchar4 c; c.x = r; c.y = g; c.z = b; c.w = a; return c;
}

// ----------------------------------------------------------------------------
// Mandelbrot-Färbung aus Iterationsbild
// Signatur muss exakt der Deklaration in nacktmull_shade.cuh entsprechen.
// ----------------------------------------------------------------------------
extern "C" __global__
void shade_from_iterations(uchar4* surface,
                           const int* __restrict__ iters,
                           int width, int height,
                           int maxIterations)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const int it  = iters[idx];

    if (it >= maxIterations) {
        // Punkte im Inneren → schwarz
        surface[idx] = pack_rgba(0u, 0u, 0u, 255u);
        return;
    }

    // Einfache, schnelle „heat“-Palette; später austauschbar (HSV etc.)
    const float t = (maxIterations > 0) ? (float)it / (float)maxIterations : 0.0f;
    const unsigned char R = (unsigned char)(255.0f * t);
    const unsigned char G = (unsigned char)(255.0f * (1.0f - t));
    const unsigned char B = (unsigned char)(255.0f * (0.5f * t));

    surface[idx] = pack_rgba(R, G, B, 255u);
}

// ----------------------------------------------------------------------------
// Debug/Diagnose: sanfter Farbverlauf + Checker-Overlay
// ----------------------------------------------------------------------------
extern "C" __global__
void shade_test_pattern(uchar4* surface,
                        int width, int height,
                        int checkSize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // Normalisierte Koordinaten ohne std::max (vermeide Header)
    const int denomW = (width  > 1) ? (width  - 1) : 1;
    const int denomH = (height > 1) ? (height - 1) : 1;
    const float u = (float)x / (float)denomW;
    const float v = (float)y / (float)denomH;

    // Basisverlauf
    float r = u;
    float g = 1.0f - 0.5f * u + 0.5f * v;
    float b = v;

    // Checker-Overlay
    const int cs = (checkSize > 0) ? checkSize : 1;
    const int cx = (x / cs) & 1;
    const int cy = (y / cs) & 1;
    const float m = ((cx ^ cy) ? 0.75f : 1.0f);
    r *= m; g *= m; b *= m;

    // clamp & pack
    r = (r < 0.f) ? 0.f : (r > 1.f ? 1.f : r);
    g = (g < 0.f) ? 0.f : (g > 1.f ? 1.f : g);
    b = (b < 0.f) ? 0.f : (b > 1.f ? 1.f : b);

    const unsigned char R = (unsigned char)(255.0f * r + 0.5f);
    const unsigned char G = (unsigned char)(255.0f * g + 0.5f);
    const unsigned char B = (unsigned char)(255.0f * b + 0.5f);

    surface[idx] = pack_rgba(R, G, B, 255u);
}
