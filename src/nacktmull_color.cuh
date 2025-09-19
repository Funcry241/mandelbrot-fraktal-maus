///// Otter: Farb-Hilfen header-only – Dither, Clamp, Luma, Vibrance/Warm-Shift, Pack sRGB8.
///// Schneefuchs: __host__ __device__ inline, keine extra TU; ASCII-only; /WX-fest; keine heavy Includes.
/// //// Maus: Kompakt (≤160 Zeilen), stabile Helfer-Namen; kompatibel zu bestehendem nacktmull.cu.
/// //// Datei: src/nacktmull_color.cuh

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
__device__ __constant__ unsigned char ZK_DITHER_BAYER8[64] = {
     0, 48, 12, 60,  3, 51, 15, 63,
    32, 16, 44, 28, 35, 19, 47, 31,
     8, 56,  4, 52, 11, 59,  7, 55,
    40, 24, 36, 20, 43, 27, 39, 23,
     2, 50, 14, 62,  1, 49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29,
    10, 58,  6, 54,  9, 57,  5, 53,
    42, 26, 38, 22, 41, 25, 37, 21
};
__device__ __forceinline__ float zk_bayer8x8_dither(int x, int y) {
    const int idx = ((y & 7) << 3) | (x & 7);
    return float(ZK_DITHER_BAYER8[idx]) * (1.0f / 64.0f) - 0.5f; // [-0.5..+0.5]
}

// ------------------------------ Farbraum-Helfer --------------------------------
__host__ __device__ __forceinline__ float zk_luma_srgb(float3 rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

// Vibrance + Warm-Shift um Luma, warmShift ~1.0 = neutral, >1 warm, <1 kalt
__host__ __device__ __forceinline__ float3 zk_vibrance_warm_shift(
    float3 rgb, float luma, float vibrance, float warmShift)
{
    // Kanalweise Verschiebung um Luma, warmShift wirkt asymmetrisch auf R/B
    const float v = vibrance;
    float3 out;
    out.x = luma + (rgb.x - luma) * v * warmShift;
    out.y = luma + (rgb.y - luma) * v * 1.0f;
    out.z = luma + (rgb.z - luma) * v * (2.0f - warmShift);
    // clamp
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
    // saturate
    r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
    g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
    b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);
    return make_uchar4((unsigned char)r, (unsigned char)g, (unsigned char)b, 255);
}
