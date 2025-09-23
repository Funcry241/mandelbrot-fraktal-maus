///// Otter: Iteration→PBO colorizer kernel – compact, branch-light; interior dark.
///// Schneefuchs: Uses Settings block geometry; stable HSV ramp with gentle gamma.
///// Maus: One kernel, one launch; clamp + pack to RGBA; no legacy includes.
///// Datei: src/colorize_iterations.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#include "settings.hpp"
#include "colorize_iterations.cuh"

// ----------------------------- tiny math helpers ------------------------------
static __device__ __forceinline__ float clamp01(float x) {
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

static __device__ __forceinline__ float3 hsv_to_rgb(float h, float s, float v) {
    // h in [0,1), s,v in [0,1]
    float r, g, b;
    float i = floorf(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    int   ii = static_cast<int>(i) % 6;
    if      (ii == 0) { r=v; g=t; b=p; }
    else if (ii == 1) { r=q; g=v; b=p; }
    else if (ii == 2) { r=p; g=v; b=t; }
    else if (ii == 3) { r=p; g=q; b=v; }
    else if (ii == 4) { r=t; g=p; b=v; }
    else              { r=v; g=p; b=q; }
    return make_float3(r, g, b);
}

static __device__ __forceinline__ uchar4 pack_rgba(float r, float g, float b, float a=1.0f) {
    r = clamp01(r); g = clamp01(g); b = clamp01(b); a = clamp01(a);
    return make_uchar4(static_cast<unsigned char>(r * 255.0f + 0.5f),
                       static_cast<unsigned char>(g * 255.0f + 0.5f),
                       static_cast<unsigned char>(b * 255.0f + 0.5f),
                       static_cast<unsigned char>(a * 255.0f + 0.5f));
}

// ------------------------------ palette mapping ------------------------------
// A smooth HSV ramp based on normalized iteration count with light easing.
// Interior pixels (it == maxIter) are rendered as near-black for "Rüsselwarze" look.
static __device__ __forceinline__ uchar4 color_from_iter(uint16_t it, int maxIter) {
    if (it >= static_cast<uint16_t>(maxIter)) {
        // Interior: keep it very dark but not pure black to preserve subtle gradients if needed
        const float v = 0.02f;
        return pack_rgba(v, v, v, 1.0f);
    }

    // Normalize and ease a bit to stretch low iterations
    const float t0 = static_cast<float>(it) / static_cast<float>(maxIter);
    const float t  = powf(t0, 0.85f); // gentle gamma

    // Hue cycles softly over a limited band to avoid rainbow noise
    const float hue = fmodf(0.62f + 0.38f * t * 1.25f, 1.0f); // 0.62..~1.0
    const float sat = 0.78f;
    const float val = 0.98f * (0.35f + 0.65f * t);            // lift with t

    const float3 rgb = hsv_to_rgb(hue, sat, val);
    return pack_rgba(rgb.x, rgb.y, rgb.z, 1.0f);
}

// ---------------------------------- kernel -----------------------------------
__global__ void kColorizeIterationsToPBO(
    const uint16_t* __restrict__ d_it,
    uchar4*       __restrict__   d_out,
    int                          width,
    int                          height,
    int                          maxIter
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int idx = y * width + x;

    const uint16_t it = d_it[idx];
    d_out[idx] = color_from_iter(it, maxIter);
}

// ---------------------------------- launch -----------------------------------
extern "C" void colorize_iterations_to_pbo(
    const uint16_t* d_iterations,
    uchar4*         d_pboOut,
    int             width,
    int             height,
    int             maxIter,
    cudaStream_t    stream
) noexcept
{
    if (!d_iterations || !d_pboOut || width <= 0 || height <= 0 || maxIter <= 0) {
        return;
    }

    dim3 block(Settings::MANDEL_BLOCK_X, Settings::MANDEL_BLOCK_Y, 1);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              1);

    kColorizeIterationsToPBO<<<grid, block, 0, stream>>>(
        d_iterations, d_pboOut, width, height, maxIter
    );
}
