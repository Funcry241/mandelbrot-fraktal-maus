///// Otter: Nacktmull-Shading - stabile Kernels; keine eigenen Typen; schnelle Helfer.
///// Schneefuchs: Deterministisch, ASCII-only; Signaturen exakt wie in .cuh deklariert; __launch_bounds__ für planbare Occupancy.
///// Maus: Kein Device-Logging; mikro-optimiert (invMax in Shared, __saturatef, rundungsfeste Pack-Funktion, __ldg, weniger Divisionen).
///// CUDA 13: nutzt __launch_bounds__(256,2) und Math-Intrinsics; Verhalten unverändert (kein Informationsverlust).
///// Datei: src/nacktmull_shade.cu

#include <cuda_runtime.h>
#include <vector_types.h>
#include "nacktmull_shade.cuh"

// --- kleine, lokale Helfer ---------------------------------------------------
static __device__ __forceinline__ float saturatef(float x) {
    // CUDA-Intrinsic: meist 1 Instruktion, clamp auf [0,1]
    return __saturatef(x);
}
static __device__ __forceinline__ uchar4 pack_rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255u) {
    uchar4 c; c.x = r; c.y = g; c.z = b; c.w = a; return c;
}

// ----------------------------------------------------------------------------
// Mandelbrot-Färbung aus Iterationsbild
// Signatur muss exakt der Deklaration in nacktmull_shade.cuh entsprechen.
// ----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256,2)
void shade_from_iterations(uchar4* __restrict__ surface,
                           const int* __restrict__ iters,
                           int width, int height,
                           int maxIterations)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Einmal pro Block: invMax vorbereiten (spart eine Division pro Thread)
    __shared__ float s_invMax;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_invMax = (maxIterations > 0) ? (1.0f / (float)maxIterations) : 0.0f;
    }
    __syncthreads();

    const int idx = y * width + x;
    const int it  = __ldg(&iters[idx]); // readonly fetch hint

    if (it >= maxIterations) {
        // Punkte im Inneren -> schwarz
        surface[idx] = pack_rgba(0u, 0u, 0u, 255u);
        return;
    }

    // Einfache, schnelle Heat-Palette: t in [0,1]
    const float t = (float)it * s_invMax;

    // Kanäle (linear gemischt, leicht austauschbar) — rundungsfest via +0.5f
    const unsigned char R = (unsigned char)(255.0f * saturatef(t)        + 0.5f);
    const unsigned char G = (unsigned char)(255.0f * saturatef(1.0f - t) + 0.5f);
    const unsigned char B = (unsigned char)(255.0f * saturatef(0.5f * t) + 0.5f);

    surface[idx] = pack_rgba(R, G, B, 255u);
}

// ----------------------------------------------------------------------------
// Debug/Diagnose: weicher Farbverlauf + Checker-Overlay
// ----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256,2)
void shade_test_pattern(uchar4* __restrict__ surface,
                        int width, int height,
                        int checkSize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Einmal pro Block: Reziproke vorbereiten (spart 2 Divisionen pro Thread)
    __shared__ float s_invW, s_invH;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const int denomW = (width  > 1) ? (width  - 1) : 1;
        const int denomH = (height > 1) ? (height - 1) : 1;
        s_invW = 1.0f / (float)denomW;
        s_invH = 1.0f / (float)denomH;
    }
    __syncthreads();

    const int idx = y * width + x;

    // Normalisierte Koordinaten ohne weitere Includes
    const float u = (float)x * s_invW;
    const float v = (float)y * s_invH;

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
