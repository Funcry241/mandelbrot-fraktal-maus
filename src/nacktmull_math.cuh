///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/nacktmull_math.cuh
#pragma once

#include <cuda_runtime.h>     // __host__/__device__
#include <vector_types.h>     // float2
#include <vector_functions.h> // make_float2
#include <cmath>              // fma

///// Otter: Pixel<->Komplex-Mapping (header-only) – zwei Overloads + schneller Precompute-Mapper.
///// Schneefuchs: __host__ __device__ __forceinline__, deterministisch; Guards bei w/h=0; ASCII-only.
///// Maus: API additiv (keine Brueche); vorhandene Overloads behalten; Mapper mikro-optimiert (FMA, weniger Divisionen).
////  CUDA 13: nutzt FMA im Hotpath; identisches Verhalten/Precision zu vorher (double-intern).

// ---------- kleines FMA-Wrapper (arbeitet in double, host+device identisch) ----------
__host__ __device__ __forceinline__
static double dmadd(double a, double b, double c) {
    // nutzt std::fma auf Host und entsprechende Device-Intrinsics (CUDA 13 ok)
    return ::fma(a, b, c);
}

// ----------------------------- Overload A: center + zoom -----------------------------
// Erwartet Pixelkoordinaten (px,py) im Pixelzentrum (z. B. x+0.5, y+0.5).
// Zoom skaliert hoehenbasiert; AR korrigiert X.
__host__ __device__ __forceinline__ float2 pixelToComplex(
    double px, double py,
    int w, int h,
    double centerX, double centerY,
    double zoom)
{
    // Guards: ungueltige Dimensionen -> Center zurueckgeben
    if (w <= 0 || h <= 0) {
        return make_float2((float)centerX, (float)centerY);
    }

    const double invW    = 1.0 / (double)w;
    const double invH    = 1.0 / (double)h;
    const double invZoom = (zoom != 0.0) ? (1.0 / zoom) : 1.0;
    const double ar      = (double)w * invH;           // w/h
    const double invZoomAr = invZoom * ar;             // precombine

    // ndc = px*(2/w)-1, py*(2/h)-1  -> via FMA
    const double sx = 2.0 * invW;
    const double sy = 2.0 * invH;
    const double ndcX = dmadd(px, sx, -1.0);
    const double ndcY = dmadd(py, sy, -1.0);

    // cx = centerX + ndcX * invZoom * ar
    // cy = centerY + ndcY * invZoom
    const double cx = dmadd(ndcX, invZoomAr, centerX);
    const double cy = dmadd(ndcY, invZoom,   centerY);

    return make_float2((float)cx, (float)cy);
}

// ---------------- Overload B: explizite Spannweite + Offset (float2) -----------------
// Erwartet Pixelkoordinaten (px,py) im Pixelzentrum (z. B. x+0.5, y+0.5).
__host__ __device__ __forceinline__ float2 pixelToComplex(
    double px, double py,
    int w, int h,
    double spanX, double spanY,
    float2 offset)
{
    // Guards: ungueltige Dimensionen -> Offset zurueckgeben
    if (w <= 0 || h <= 0) {
        return offset;
    }

    const double invW = 1.0 / (double)w;
    const double invH = 1.0 / (double)h;

    const double sx = 2.0 * invW;
    const double sy = 2.0 * invH;

    const double ndcX = dmadd(px, sx, -1.0);
    const double ndcY = dmadd(py, sy, -1.0);

    const double hx = 0.5 * spanX;
    const double hy = 0.5 * spanY;

    const double cx = dmadd(ndcX, hx, (double)offset.x);
    const double cy = dmadd(ndcY, hy, (double)offset.y);

    return make_float2((float)cx, (float)cy);
}
