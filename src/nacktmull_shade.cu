///// MAUS: implementation for Nacktmull shading kernel (no uchar4 redefinition)
#include <cuda_runtime.h>
#include <vector_types.h>
#include "nacktmull_shade.cuh"

// 🦦 Otter: Keine eigene uchar4-Struct! Wir verwenden die aus <vector_types.h>. (Bezug zu Otter)
// 🦊 Schneefuchs: Einfaches, deterministisches Farbschema; Host/Device-Logging bleibt aus. (Bezug zu Schneefuchs)

__device__ __forceinline__ uchar4 pack_rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255) {
    uchar4 c; c.x = r; c.y = g; c.z = b; c.w = a; return c;
}

extern "C" __global__
void shade_from_iterations(uchar4* rgba, const int* iterations,
                           int width, int height, int maxIter)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const int it  = iterations[idx];

    uchar4 c;
    if (it >= maxIter) {
        // Fraktalkern dunkel
        c = pack_rgba(0u, 0u, 0u, 255u);
    } else {
        // Linearer Verlauf 0..1
        const float t = (maxIter > 0) ? (static_cast<float>(it) / static_cast<float>(maxIter)) : 0.0f;

        // Einfache, schnelle “heat”-Mischung (State of the Art genug für Start, leicht austauschbar):
        // R = t, G = 1-t, B = 0.5*t  -> später gern durch Warzen-HSV ersetzen.
        const unsigned char R = static_cast<unsigned char>(255.0f * t);
        const unsigned char G = static_cast<unsigned char>(255.0f * (1.0f - t));
        const unsigned char B = static_cast<unsigned char>(255.0f * (0.5f * t));
        c = pack_rgba(R, G, B, 255u);
    }

    rgba[idx] = c;
}
