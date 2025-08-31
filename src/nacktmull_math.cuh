///// Otter: Pixel<->Komplex-Mapping (header-only) – zwei Overloads + schneller Precompute-Mapper.
///// Schneefuchs: __host__ __device__ __forceinline__, deterministisch; Guards bei w/h=0; ASCII-only.
///// Maus: API additiv (keine Brueche); vorhandene Overloads behalten; neuer Mapper fuer Performance.

#pragma once

#include <cuda_runtime.h>   // __host__/__device__
#include <vector_types.h>   // float2
#include <vector_functions.h> // make_float2

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
        return make_float2(static_cast<float>(centerX), static_cast<float>(centerY));
    }

    const double invW    = 1.0 / static_cast<double>(w);
    const double invH    = 1.0 / static_cast<double>(h);
    const double invZoom = (zoom != 0.0) ? (1.0 / zoom) : 1.0;
    const double ar      = static_cast<double>(w) * invH;

    const double ndcX = px * (2.0 * invW) - 1.0;
    const double ndcY = py * (2.0 * invH) - 1.0;

    const double cx = centerX + ndcX * invZoom * ar;
    const double cy = centerY + ndcY * invZoom;

    return make_float2(static_cast<float>(cx), static_cast<float>(cy));
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

    const double invW = 1.0 / static_cast<double>(w);
    const double invH = 1.0 / static_cast<double>(h);

    const double ndcX = px * (2.0 * invW) - 1.0;
    const double ndcY = py * (2.0 * invH) - 1.0;

    const double halfX = spanX * 0.5;
    const double halfY = spanY * 0.5;

    const double cx = static_cast<double>(offset.x) + ndcX * halfX;
    const double cy = static_cast<double>(offset.y) + ndcY * halfY;

    return make_float2(static_cast<float>(cx), static_cast<float>(cy));
}

// ------------------------------ Schneller Precompute-Mapper -------------------------
// Fuer Hot-Loops (Kernel/CPU) ohne wiederholte Divisionen: einmal Faktoren berechnen,
// dann operator()(x+0.5, y+0.5) aufrufen. API ist additiv – bestehende Aufrufe bleiben gueltig.
struct PixelToComplexMapZoom {
    double cx, cy, invZoom, ar, sx, sy; // sx=2/w, sy=2/h
    int    valid;                        // 1 wenn w>0 && h>0
    __host__ __device__ __forceinline__
    PixelToComplexMapZoom(int w, int h, double centerX, double centerY, double zoom)
    : cx(centerX), cy(centerY),
      invZoom(zoom != 0.0 ? (1.0/zoom) : 1.0),
      ar( (h > 0) ? (static_cast<double>(w)/static_cast<double>(h)) : 1.0 ),
      sx( (w > 0) ? (2.0/static_cast<double>(w)) : 2.0 ),
      sy( (h > 0) ? (2.0/static_cast<double>(h)) : 2.0 ),
      valid( (w > 0 && h > 0) ? 1 : 0 )
    {}
    __host__ __device__ __forceinline__
    float2 operator()(double px, double py) const {
        if (!valid) {
            return make_float2(static_cast<float>(cx), static_cast<float>(cy));
        }
        const double ndcX = px * sx - 1.0;
        const double ndcY = py * sy - 1.0;
        const double cx_  = cx + ndcX * invZoom * ar;
        const double cy_  = cy + ndcY * invZoom;
        return make_float2(static_cast<float>(cx_), static_cast<float>(cy_));
    }
};

struct PixelToComplexMapSpan {
    double ox, oy, hx, hy, sx, sy; // hx=spanX/2, hy=spanY/2, sx=2/w, sy=2/h
    int    valid;                   // 1 wenn w>0 && h>0
    __host__ __device__ __forceinline__
    PixelToComplexMapSpan(int w, int h, double spanX, double spanY, float2 offset)
    : ox(static_cast<double>(offset.x)), oy(static_cast<double>(offset.y)),
      hx(spanX * 0.5), hy(spanY * 0.5),
      sx( (w > 0) ? (2.0/static_cast<double>(w)) : 2.0 ),
      sy( (h > 0) ? (2.0/static_cast<double>(h)) : 2.0 ),
      valid( (w > 0 && h > 0) ? 1 : 0 )
    {}
    __host__ __device__ __forceinline__
    float2 operator()(double px, double py) const {
        if (!valid) {
            return make_float2(static_cast<float>(ox), static_cast<float>(oy));
        }
        const double ndcX = px * sx - 1.0;
        const double ndcY = py * sy - 1.0;
        const double cx_  = ox + ndcX * hx;
        const double cy_  = oy + ndcY * hy;
        return make_float2(static_cast<float>(cx_), static_cast<float>(cy_));
    }
};
