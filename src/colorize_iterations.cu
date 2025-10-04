///// Otter: Iteration→PBO colorizer – fine gradients via local iteration gradient (phase blend).
///// Schneefuchs: Cosine palette, gamma-eased, deterministic; no API changes.
/**  Maus: Interior stays dark with thin halo; stable & flicker-free; only this TU changed.  */
///// Datei: src/colorize_iterations.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#include "settings.hpp"
#include "colorize_iterations.cuh"

// ----------------------------- tiny math helpers ------------------------------
static __device__ __forceinline__ float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static __device__ __forceinline__ uchar4 pack_rgba(float r, float g, float b, float a=1.0f) {
    r = clamp01(r); g = clamp01(g); b = clamp01(b); a = clamp01(a);
    return make_uchar4((unsigned char)(r * 255.0f + 0.5f),
                       (unsigned char)(g * 255.0f + 0.5f),
                       (unsigned char)(b * 255.0f + 0.5f),
                       (unsigned char)(a * 255.0f + 0.5f));
}

// deterministic 32→[0,1) hash (PCG-ish mix; fixed, frame-stable)
static __device__ __forceinline__ float hash01(uint32_t x){
    x ^= x >> 17; x *= 0xed5ad4bbu;
    x ^= x >> 11; x *= 0xac4c1b51u;
    x ^= x >> 15; x *= 0x31848babu;
    x ^= x >> 14;
    return (x >> 8) * (1.0f / 16777216.0f); // use top 24 bits
}

// Inigo-Quílez-artige Cosine-Palette
static __device__ __forceinline__ float3 cosine_palette(float t, float3 a, float3 b, float3 c, float3 d) {
    const float twoPi = 6.283185307179586f;
    float3 ct = make_float3(c.x * t + d.x, c.y * t + d.y, c.z * t + d.z);
    return make_float3(a.x + b.x * cosf(twoPi * ct.x),
                       a.y + b.y * cosf(twoPi * ct.y),
                       a.z + b.z * cosf(twoPi * ct.z));
}

// ---------------------------- tunables (band smoothing) -----------------------
static __constant__ float kPHASE_GRAD = 0.12f; // Anteil Phase aus lokalem Gradienten (0.08..0.18)
static __constant__ float kPHASE_HASH = 0.03f; // geringe, statische Pixelphase gegen Restbanding

// -------------------------------- palette map --------------------------------
// Innen bleibt dunkel; außen Cosine-Palette. Bänder werden über eine
// phasenstabile, ortsgebundene Verschiebung (Gradient+Hash) geglättet.
static __device__ __forceinline__ uchar4 color_from_iter_ex(
    uint16_t it, int maxIter, int idxLinear, float grad01)
{
    if (maxIter <= 1) { const float v = 0.02f; return pack_rgba(v,v,v,1.0f); }

    const int interiorEdge = max(0, maxIter - 1);
    const int haloWidth    = 6;

    // Innenbereich sehr dunkel
    if ((int)it >= interiorEdge) {
        const float v = 0.015f;
        return pack_rgba(v,v,v,1.0f);
    }

    // Normierung mit leichter Entzerrung + Gradientenantail (sub-iter)
    float t0 = ((float)it + 0.65f * grad01) / (float)max(interiorEdge, 1);
    t0 = clamp01(t0);
    float t  = powf(t0, 0.82f);

    // Mehr Varianz ohne harte Bänder, Zyklen über 0..1
    const float cycles = 2.90f;

    // Phasenverschiebung: lokal (Gradient) + minimale statische Pixelphase
    const float phi = kPHASE_GRAD * grad01
                    + kPHASE_HASH * (hash01((uint32_t)(idxLinear * 747796405u)) - 0.5f);
    float k = t * cycles + phi; k -= floorf(k);

    // Cosine-Palette-Parameter (fein abgestimmt)
    const float3 A = make_float3(0.52f, 0.46f, 0.50f);
    const float3 B = make_float3(0.48f, 0.42f, 0.46f);
    const float3 C = make_float3(1.00f, 1.00f, 1.00f);
    const float3 D = make_float3(0.00f, 0.18f, 0.38f);

    float3 col = cosine_palette(k, A, B, C, D);

    // Heller Saum kurz vor innen
    const int toEdge = interiorEdge - (int)it; // 1..haloWidth
    if (toEdge > 0 && toEdge <= haloWidth) {
        const float s = (float)(haloWidth - toEdge + 1) / (float)haloWidth; // 0..1
        const float boost = 0.18f * s;
        col.x = clamp01(col.x + boost);
        col.y = clamp01(col.y + boost);
        col.z = clamp01(col.z + boost);
    }

    // leichte Gamma auf Value für knackigere Lichter
    const float gamma = 0.92f;
    col.x = powf(col.x, gamma);
    col.y = powf(col.y, gamma);
    col.z = powf(col.z, gamma);

    return pack_rgba(col.x, col.y, col.z, 1.0f);
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

    const uint16_t it  = d_it[idx];

    // sehr billiger lokaler Gradient (vorwärts, frame-stabil, O(1))
    const int xr = min(x + 1, width  - 1);
    const int yd = min(y + 1, height - 1);
    const uint16_t itR = d_it[y  * width + xr];
    const uint16_t itD = d_it[yd * width + x ];

    const float gx = (float)((int)itR - (int)it);
    const float gy = (float)((int)itD - (int)it);
    float grad = sqrtf(gx*gx + gy*gy);

    // auf 0..1 normieren (empirisch, verhindert Übersteuerung)
    float grad01 = grad * (1.0f / 6.0f);
    if (grad01 > 1.0f) grad01 = 1.0f;

    d_out[idx] = color_from_iter_ex(it, maxIter, idx, grad01);
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
