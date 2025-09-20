///// Otter: sRGB-Pack & Farbhefler; header-only.
///// Schneefuchs: Keine Device-Globals; reine Funktionen; /WX-fest.
///// Maus: Performantes Packen; minimale Zweige.
///// Datei: src/nacktmull_color.cuh

#pragma once

// Hinweis: Header-only, wird in .cu inkludiert. Keine externen TU-Symbole.
// Keine Includes nötig; nutzt nur CUDA Builtins und Primitive.

// -------------------------- Math-Utilities (inline) ---------------------------
__host__ __device__ __forceinline__ float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
__host__ __device__ __forceinline__ float mixf(float a, float b, float t) {
    return a + t * (b - a);
}
__host__ __device__ __forceinline__ float fractf(float x) {
    return x - floorf(x);
}

// ------------------------------ Dither (Bayer 8x8) ----------------------------
// Liefert exakt die Standard-8x8-Bayer-Matrix (0..63) ohne globale Tabelle.
// Mapping basiert auf rekursiver 2x2-Bayer-Expansion; Level-Reihenfolge: LSB→MSB.
__device__ __forceinline__ int zk_bayer8x8_index(int x, int y) {
    int v = 0;
    #pragma unroll
    for (int level = 0; level < 3; ++level) { // 2^3 = 8
        const int bx = (x >> level) & 1;
        const int by = (y >> level) & 1;
        v <<= 2;
        // Codes für 2x2-Basis: (0,0)->0, (1,0)->3, (0,1)->2, (1,1)->1
        v += (bx == 0 && by == 0) ? 0
           : (bx == 1 && by == 0) ? 3
           : (bx == 0 && by == 1) ? 2
           :                         1;
    }
    return v; // 0..63
}
__device__ __forceinline__ float zk_bayer8x8_dither(int x, int y) {
    return float(zk_bayer8x8_index(x, y)) * (1.0f / 64.0f) - 0.5f; // [-0.5..+0.5]
}

// ------------------------------ Farbraum-Helfer --------------------------------
__host__ __device__ __forceinline__ float zk_luma_srgb(float3 rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

// Vibrance + Warm-Shift um Luma, warmShift ~1.0 = neutral, >1 warm, <1 kalt
__host__ __device__ __forceinline__ float3 zk_vibrance_warm_shift(
    float3 rgb, float luma, float vibrance, float warmShift)
{
    const float v = vibrance;
    float3 out;
    out.x = luma + (rgb.x - luma) * v * warmShift;
    out.y = luma + (rgb.y - luma) * v * 1.0f;
    out.z = luma + (rgb.z - luma) * v * (2.0f - warmShift);
    out.x = clamp01(out.x); out.y = clamp01(out.y); out.z = clamp01(out.z);
    return out;
}

// --------------------------------- Pack/Compose --------------------------------
// sRGB8 Pack mit Bayer-Dither; Alpha=255
__device__ __forceinline__ uchar4 zk_pack_srgb8(float3 rgb, int px, int py) {
    const float d = zk_bayer8x8_dither(px, py);
    float r = fmaf(255.0f, clamp01(rgb.x), 0.5f + d);
    float g = fmaf(255.0f, clamp01(rgb.y), 0.5f + d);
    float b = fmaf(255.0f, clamp01(rgb.z), 0.5f + d);
    r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
    g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
    b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);
    return make_uchar4((unsigned char)r, (unsigned char)g, (unsigned char)b, 255);
}
