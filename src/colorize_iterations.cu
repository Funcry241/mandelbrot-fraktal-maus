///// Otter: Iteration→PBO colorizer – richer background (deep-space gradient + subtle dither).
///// Schneefuchs: Bandarm (cosine palette), gamma-eased, deterministic hash; no API changes.
///// Maus: Interior stays dark with thin halo; outside gets lively but not noisy.
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
static __device__ __forceinline__ float3 lerp3(const float3& a, const float3& b, float t){
    return make_float3(a.x + (b.x - a.x)*t,
                       a.y + (b.y - a.y)*t,
                       a.z + (b.z - a.z)*t);
}

// deterministic 32→[0,1) hash (PCG-ish mix; fixed)
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

// ---------------------------- tunables (background) ---------------------------
// Unterhalb dieses Bruchs (rel. Iteration) behandeln wir "Hintergrund".
static __constant__ float kBG_SPLIT = 0.08f;   // 0..1 — Anteil der Iterationsspanne für „Low-Detail“
static __constant__ float kBG_NOISE = 0.03f;   // max. Dither-Amplitude (nur im Hintergrundzweig)
static __constant__ float kBG_BLEND = 0.28f;   // Anteil der Cosine-Farbkomponente im Hintergrund

// ------------------------------ palette mapping ------------------------------
// Innen bleibt dunkel; außen lebendige Cosine-Palette.
// Für sehr niedrige Iterationen (Hintergrund) gibt es einen „Deep-Space“-Verlauf
// + leichtes Dithering, damit große Flächen nicht eintönig blau wirken.
static __device__ __forceinline__ uchar4 color_from_iter(
    uint16_t it, int maxIter, int idxLinear)
{
    if (maxIter <= 1) {
        const float v = 0.02f; return pack_rgba(v, v, v, 1.0f);
    }

    const int interiorEdge = max(0, maxIter - 1);
    const int haloWidth    = 6;

    // Innenbereich sehr dunkel
    if ((int)it >= interiorEdge) {
        const float v = 0.015f;
        return pack_rgba(v, v, v, 1.0f);
    }

    // Normierung
    float t0 = (float)it / (float)max(interiorEdge, 1);
    float t  = powf(t0, 0.82f); // leichte Entzerrung

    // ---------- Hintergrund-Behandlung ----------
    if (t0 < kBG_SPLIT) {
        // u skaliert die Subspanne [0..kBG_SPLIT] → [0..1]
        const float u = (kBG_SPLIT > 1e-6f) ? (t0 / kBG_SPLIT) : 0.f;

        // Deep-Space-Verlauf (kohlig → kühles Indigo)
        const float3 spaceA = make_float3(0.06f, 0.07f, 0.09f);
        const float3 spaceB = make_float3(0.10f, 0.11f, 0.16f);
        float3 bg = lerp3(spaceA, spaceB, powf(u, 1.35f));

        // Etwas Farbleben via Cosine-Palette, schwach beigemischt
        // (begrenzter Hue-Sweep, damit keine „Regenbogen“-Anmutung)
        const float3 A = make_float3(0.50f, 0.46f, 0.52f);
        const float3 B = make_float3(0.36f, 0.30f, 0.34f);
        const float3 C = make_float3(1.00f, 1.00f, 1.00f);
        const float3 D = make_float3(0.05f, 0.22f, 0.40f);
        const float  cycles = 1.35f;
        float k = t * cycles - floorf(t * cycles); // fract
        float3 cosCol = cosine_palette(k, A, B, C, D);

        // Mischung (mehr „space“ bei u≈0, mehr Cosine wenn u wächst)
        const float mixAmt = kBG_BLEND * (0.35f + 0.65f * u);
        float3 col = lerp3(bg, cosCol, mixAmt);

        // Sehr feines, deterministisches Dithering, das mit u ausfadet
        float jitter = (hash01((uint32_t)idxLinear * 1664525u) - 0.5f) * (kBG_NOISE * (1.0f - u));
        col.x = clamp01(col.x + jitter);
        col.y = clamp01(col.y + jitter);
        col.z = clamp01(col.z + jitter);

        // leichte Gamma auf Value
        const float gamma = 0.95f;
        col.x = powf(col.x, gamma);
        col.y = powf(col.y, gamma);
        col.z = powf(col.z, gamma);

        return pack_rgba(col.x, col.y, col.z, 1.0f);
    }

    // ---------- „Detail“-Palette (außerhalb Hintergrund) ----------
    // Mehr Varianz: mehrere Zyklen über 0..1 (ohne harte Bänder)
    const float cycles = 3.25f;
    float k = t * cycles - floorf(t * cycles);

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

    const uint16_t it = d_it[idx];
    d_out[idx] = color_from_iter(it, maxIter, idx);
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
