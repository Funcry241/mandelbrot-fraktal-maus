///// MAUS: pixel <-> complex mapping helpers (header-only, dual overload)
#pragma once
// 🦊 Schneefuchs: Leichtgewichtig, nur CUDA-Basics. (Bezug zu Schneefuchs)
// 🦦 Otter: Overload für (spanX,spanY,offset) wie in nacktmull.cu aufgerufen. (Bezug zu Otter)

#include <vector_types.h>   // float2
#include <cuda_runtime.h>   // make_float2

// Overload A: center + zoom
__host__ __device__ inline float2 pixelToComplex(
    double px, double py,
    int w, int h,
    double centerX, double centerY,
    double zoom)
{
    const double invZoom = (zoom != 0.0) ? (1.0 / zoom) : 1.0;
    const double ar = (h != 0) ? static_cast<double>(w) / static_cast<double>(h) : 1.0;

    const double ndcX = (px / static_cast<double>(w)) * 2.0 - 1.0;
    const double ndcY = (py / static_cast<double>(h)) * 2.0 - 1.0;

    const double cx = centerX + ndcX * invZoom * ar;
    const double cy = centerY + ndcY * invZoom;
    return make_float2(static_cast<float>(cx), static_cast<float>(cy));
}

// Overload B: explizite Spannweite + Offset (Fix für deinen Build)
__host__ __device__ inline float2 pixelToComplex(
    double px, double py,
    int w, int h,
    double spanX, double spanY,
    float2 offset)
{
    const double ndcX = (px / static_cast<double>(w)) * 2.0 - 1.0;
    const double ndcY = (py / static_cast<double>(h)) * 2.0 - 1.0;

    const double cx = static_cast<double>(offset.x) + ndcX * (spanX * 0.5);
    const double cy = static_cast<double>(offset.y) + ndcY * (spanY * 0.5);
    return make_float2(static_cast<float>(cx), static_cast<float>(cy));
}
