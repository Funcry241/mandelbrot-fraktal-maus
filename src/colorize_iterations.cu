///// Otter: Nacktmull - Stage-1 "Cool Tech" palette (cyan->steel->indigo), micro-contrast; perf-safe; no API change
///// Schneefuchs: Cosine palette with grad+hash+Bayer phase; sqrt via rsqrtf; stripes; ASCII-only
///// Maus: Interior dark, thinner flats; deterministic; only this TU adjusted
///// Datei: src/colorize_iterations.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#include "settings.hpp"
#include "colorize_iterations.cuh"

// ----------------------------- tiny math helpers ------------------------------
static __device__ __forceinline__ float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static __device__ __forceinline__ float lerpf(float a, float b, float t){ return a + t * (b - a); }
static __device__ __forceinline__ uchar4 pack_rgba(float r, float g, float b, float a=1.0f) {
    r = clamp01(r); g = clamp01(g); b = clamp01(b); a = clamp01(a);
    return make_uchar4((unsigned char)(r * 255.0f + 0.5f),
                       (unsigned char)(g * 255.0f + 0.5f),
                       (unsigned char)(b * 255.0f + 0.5f),
                       (unsigned char)(a * 255.0f + 0.5f));
}
// integer max without pulling <algorithm> into this TU
static __device__ __forceinline__ int i_max(int a, int b) { return a > b ? a : b; }

// deterministic 32->[0,1) hash (PCG-ish mix; fixed, frame-stable)
static __device__ __forceinline__ float hash01(uint32_t x){
    x ^= x >> 17; x *= 0xed5ad4bbu;
    x ^= x >> 11; x *= 0xac4c1b51u;
    x ^= x >> 15; x *= 0x31848babu;
    x ^= x >> 14;
    return (x >> 8) * (1.0f / 16777216.0f); // use top 24 bits
}

// Inigo-Quilez-like cosine palette
static __device__ __forceinline__ float3 cosine_palette(float t, float3 a, float3 b, float3 c, float3 d) {
    const float twoPi = 6.283185307179586f;
    float3 ct = make_float3(c.x * t + d.x, c.y * t + d.y, c.z * t + d.z);
    return make_float3(a.x + b.x * cosf(twoPi * ct.x),
                       a.y + b.y * cosf(twoPi * ct.y),
                       a.z + b.z * cosf(twoPi * ct.z));
}

// ---------------------------- tunables (band smoothing) -----------------------
static __constant__ float kPHASE_GRAD   = 0.12f;   // phase from local gradient (0.08..0.18)
static __constant__ float kPHASE_HASH   = 0.025f;  // tiny static per-pixel phase
static __constant__ float kPHASE_ORIENT = 0.030f;  // small orientation phase from gradient direction
static __constant__ float kV_GAIN       = 0.05f;   // micro-contrast on Value (0..~0.08)  [Cool Tech]
static __constant__ float kGRAD_NORM    = 1.0f / 6.0f; // empirical gradient normalization
static __constant__ float kBIAS_MIX     = 0.35f;   // mix toward sqrt(x) ~ gamma 0.82
static __constant__ float kCYCLES       = 3.80f;   // slightly higher cycle density       [Cool Tech]
static __constant__ float kSTRIPE_FREQ  = 0.85f;   // stripe frequency on t
static __constant__ float kSTRIPE_GAIN  = 0.07f;   // brightness modulation via stripes
static __constant__ float kBAYER_AMP    = 0.020f;  // 4x4 Bayer phase amplitude (~2% of a cycle)

// 4x4 Bayer (0..15), in order (x + 4*y)
__device__ __constant__ unsigned char kBayer4x4[16] = {
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5
};

// fast length approx ~ sqrt(x^2+y^2) (max + 0.375*min) — avoids sqrtf
static __device__ __forceinline__ float fast_len2(float ax, float ay){
    ax = fabsf(ax); ay = fabsf(ay);
    const float m = fmaxf(ax, ay);
    const float n = fminf(ax, ay);
    return m + 0.375f * n;
}

// fast sqrt for [0,1]: x*rsqrt(x), stabilized with eps
static __device__ __forceinline__ float fast_sqrt01(float x){
    x = (x <= 0.f) ? 0.f : x;
    const float e = fmaxf(x, 1e-8f);
    return x * rsqrtf(e);
}

// -------------------------------- palette map --------------------------------
// Interior stays dark; exterior uses cosine palette. Banding is reduced by a
// stable phase shift (gradient + hash + Bayer). Added micro-contrast and stripes.
static __device__ __forceinline__ uchar4 color_from_iter_ex(
    uint16_t it, int maxIter, int idxLinear,
    float grad01, float orient, int px, int py)
{
    if (maxIter <= 1) { const float v = 0.02f; return pack_rgba(v,v,v,1.0f); }

    const int interiorEdge = i_max(0, maxIter - 1);
    const int haloWidth    = 6;

    // interior: very dark
    if ((int)it >= interiorEdge) {
        const float v = 0.015f;
        return pack_rgba(v,v,v,1.0f);
    }

    // normalize and bias toward sqrt (no powf)
    float t0 = ((float)it + 0.65f * grad01) / (float)i_max(interiorEdge, 1);
    t0 = clamp01(t0);
    float t  = lerpf(t0, fast_sqrt01(t0), kBIAS_MIX); // ~ x^0.82

    // phase = grad + tiny hash + small orientation + 4x4 Bayer
    const float hashP   = hash01((uint32_t)(idxLinear * 747796405u)) - 0.5f; // [-0.5,0.5)
    const int   bIdx    = (px & 3) | ((py & 3) << 2);
    const float bayerP  = ((float)kBayer4x4[bIdx] * (1.0f/15.0f)) - 0.5f;    // [-0.5,0.5]
    const float phi     = kPHASE_GRAD * grad01
                        + kPHASE_HASH * hashP
                        + kPHASE_ORIENT * orient
                        + kBAYER_AMP   * bayerP;

    float k = t * kCYCLES + phi;
    k -= floorf(k);

    // Cool Tech palette (cyan -> steel -> indigo)
    const float3 A = make_float3(0.30f, 0.36f, 0.42f);
    const float3 B = make_float3(0.34f, 0.38f, 0.50f);
    const float3 C = make_float3(1.00f, 1.00f, 1.00f);
    const float3 D = make_float3(0.05f, 0.22f, 0.62f);

    float3 col = cosine_palette(k, A, B, C, D);

    // thin bright rim near interior (branch-light)
    const int toEdge = interiorEdge - (int)it; // 1..haloWidth
    if (toEdge > 0 && toEdge <= haloWidth) {
        const float s = (float)(haloWidth - toEdge + 1) / (float)haloWidth; // 0..1
        const float boost = 0.18f * s;
        col.x = clamp01(col.x + boost);
        col.y = clamp01(col.y + boost);
        col.z = clamp01(col.z + boost);
    }

    // micro-contrast on Value via S-curve of grad01
    const float curve = grad01 * (2.0f - grad01);
    const float vGain = 1.0f + kV_GAIN * (curve - 0.5f); // symmetric around 1.0
    col.x = clamp01(col.x * vGain);
    col.y = clamp01(col.y * vGain);
    col.z = clamp01(col.z * vGain);

    // stripe modulation on t (prevents large flat areas)
    const float stripe = 0.5f + 0.5f * cosf(6.283185307179586f * (t * kSTRIPE_FREQ + 0.5f*phi));
    const float sGain  = 1.0f + kSTRIPE_GAIN * (stripe - 0.5f);
    col.x = clamp01(col.x * sGain);
    col.y = clamp01(col.y * sGain);
    col.z = clamp01(col.z * sGain);

    return pack_rgba(col.x, col.y, col.z, 1.0f);
}

// ---------------------------------- kernel -----------------------------------
__global__ __launch_bounds__(Settings::MANDEL_BLOCK_X * Settings::MANDEL_BLOCK_Y, 2)
void kColorizeIterationsToPBO(
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

    const uint16_t it  = d_it[idx];

    // very cheap local gradient (forward, frame-stable, O(1))
    const int xr = min(x + 1, width  - 1);
    const int yd = min(y + 1, height - 1);
    const uint16_t itR = d_it[y  * width + xr];
    const uint16_t itD = d_it[yd * width + x ];

    const float gx = (float)((int)itR - (int)it);
    const float gy = (float)((int)itD - (int)it);

    // fast length instead of sqrtf(gx*gx+gy*gy)
    float grad = fast_len2(gx, gy);

    // normalize to 0..1 (empirical, avoids overdrive)
    float grad01 = grad * kGRAD_NORM;
    if (grad01 > 1.0f) grad01 = 1.0f;

    // coarse orientation phase from gradient direction (no atan2f)
    float denom = fabsf(gx) + fabsf(gy) + 1e-6f;
    float orient = (denom > 0.f) ? (gx / denom) : 0.f;

    d_out[idx] = color_from_iter_ex(it, maxIter, idx, grad01, orient, x, y);
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
    if (!d_iterations || !d_pboOut || width <= 0 || height <= 0 || maxIter <= 0) return;

    dim3 block(Settings::MANDEL_BLOCK_X, Settings::MANDEL_BLOCK_Y, 1);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              1);

    kColorizeIterationsToPBO<<<grid, block, 0, stream>>>(d_iterations, d_pboOut, width, height, maxIter);
}
