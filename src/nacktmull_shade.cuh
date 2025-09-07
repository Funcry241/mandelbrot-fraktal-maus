#pragma once

// CUDA / Vektor-Typen
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h> // make_float3, make_uchar4

// Standard-Math
#include <cmath>

// ------------------------ kleine Device-Helfer (CUDA 13, mikro-optimiert) ----
#if defined(__CUDACC__)
static __device__ __forceinline__ float clamp01(float x) {
    // CUDA-Intrinsic: eine Instruktion, exakt [0,1] clamp
    return __saturatef(x);
}

static __device__ __forceinline__ uchar4 make_rgba_01(float r, float g, float b, float a = 1.f) {
    r = __saturatef(r); g = __saturatef(g); b = __saturatef(b); a = __saturatef(a);
    return make_uchar4(
        (unsigned char)(r * 255.f + 0.5f),
        (unsigned char)(g * 255.f + 0.5f),
        (unsigned char)(b * 255.f + 0.5f),
        (unsigned char)(a * 255.f + 0.5f)
    );
}

// HSV→RGB: h∈[0,1), s,v∈[0,1]; mit FMA-Formulierung für p/q/t
static __device__ __forceinline__ float3 hsv2rgb(float h, float s, float v) {
    float r = v, g = v, b = v;
    if (s > 0.f) {
        // h6 in [0,6)
        const float hf = h - floorf(h);     // sauberer Wrap ohne benötigte intrinsics
        const float h6 = hf * 6.f;
        const int   i  = (int)h6;           // 0..5
        const float f  = h6 - (float)i;

        const float vs = v * s;
        const float p  = v - vs;
        const float q  = __fmaf_rn(-vs, f, v);       // v - vs*f
        const float t  = __fmaf_rn(vs, f, p);        // p + vs*f

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

// ------------------------ Kernel-Prototypen (Signaturen stabil) --------------
#if defined(__CUDACC__)
extern "C" __global__ __launch_bounds__(256,2)
void shade_from_iterations(uchar4* __restrict__ surface,
                           const int* __restrict__ iters,
                           int width, int height,
                           int maxIterations);

extern "C" __global__ __launch_bounds__(256,2)
void shade_test_pattern(uchar4* __restrict__ surface,
                        int width, int height,
                        int checkSize);
#endif // __CUDACC__
