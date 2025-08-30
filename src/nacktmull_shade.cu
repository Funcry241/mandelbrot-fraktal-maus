///// Otter: Nacktmull-Shading – stabile Kernels; keine eigenen Typen; schnelle Helfer.
///// Schneefuchs: Deterministisch, ASCII-only; Signaturen exakt wie in .cuh deklariert.
///// Maus: Kein Device-Logging; mikro-optimiert (invMax, saturatef, rundungsfeste Pack-Funktion).

#include <cuda_runtime.h>
#include <vector_types.h>
#include "nacktmull_shade.cuh"

// --- kleine, lokale Helfer ---------------------------------------------------
static __device__ __forceinline__ float saturatef(float x) {
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}
static __device__ __forceinline__ uchar4 pack_rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255u) {
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

    // Einfache, schnelle Heat-Palette: t in [0,1]
    const float invMax = (maxIterations > 0) ? (1.0f / (float)maxIterations) : 0.0f;
    const float t = (float)it * invMax;

    // Kanäle (linear gemischt, später leicht austauschbar)
    const unsigned char R = (unsigned char)(255.0f * saturatef(t) + 0.5f);
    const unsigned char G = (unsigned char)(255.0f * saturatef(1.0f - t) + 0.5f);
    const unsigned char B = (unsigned char)(255.0f * saturatef(0.5f * t) + 0.5f);

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

    // Normalisierte Koordinaten ohne std::max (vermeide zusätzliche Includes)
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

    // clamp & pack (rundungsfest)
    const unsigned char R = (unsigned char)(255.0f * saturatef(r) + 0.5f);
    const unsigned char G = (unsigned char)(255.0f * saturatef(g) + 0.5f);
    const unsigned char B = (unsigned char)(255.0f * saturatef(b) + 0.5f);

    surface[idx] = pack_rgba(R, G, B, 255u);
}
