///// Otter: Pixel→Complex mapping hygiene for Capybara (Hi/Lo + exact step split).
///// Schneefuchs: Header-only; safe include anywhere; ASCII-only; rate-limited device logs via capy_* helpers.
///// Maus: Returns both classic double2 and CapyHiLo2; no API break for existing kernels.
///// Datei: src/capybara_mapping.cuh

#pragma once
#include <math.h>
#include <stdint.h>

#include "capybara_math.cuh" // CapyHiLo/CapyHiLo2 + helpers

#if defined(__CUDACC__)
#define CAPY_HD __host__ __device__ __forceinline__
#define CAPY_D  __device__ __forceinline__
#else
#define CAPY_HD inline
#define CAPY_D  inline
#endif

// ----------------------------- GID & pixel offsets ----------------------------
CAPY_HD uint32_t capy_gid(int x, int y, int w)
{
    // Non-negative by construction (kernel guards ensure ranges).
    return (uint32_t)(y * w + x);
}

CAPY_HD void capy_pixel_offsets(int px, int py, int w, int h, double& offx, double& offy)
{
    // Center-of-pixel convention: (+0.5) then subtract image center (w/2, h/2)
    offx = (double(px) + 0.5) - 0.5 * double(w);
    offy = (double(py) + 0.5) - 0.5 * double(h);
}

// ----------------------------- Classic mapping (double) -----------------------
CAPY_HD double2 capy_map_pixel_double(double cx, double cy,
                                      double stepX, double stepY,
                                      int px, int py, int w, int h)
{
    double offx, offy;
    capy_pixel_offsets(px, py, w, h, offx, offy);
    double2 c;
    c.x = cx + offx * stepX;
    c.y = cy + offy * stepY;
    return c;
}

// ----------------------------- Capybara mapping (Hi/Lo) -----------------------
CAPY_HD CapyHiLo2 capy_map_pixel_hilo(double cx, double cy,
                                      double stepX, double stepY,
                                      int px, int py, int w, int h)
{
    double offx, offy;
    capy_pixel_offsets(px, py, w, h, offx, offy);
    // Uses frexp/ldexp split internally to keep step structure at deep zooms,
    // and compensated addition for center + delta.
    return capy_map_center_step(cx, cy, stepX, stepY, offx, offy);
}

// ----------------------------- Bridging helper --------------------------------
// Fills both classic and Hi/Lo coordinates so existing kernels can adopt incrementally.
// If you only need one, call the specialized functions above.
CAPY_HD void capy_map_pixel(double cx, double cy,
                            double stepX, double stepY,
                            int px, int py, int w, int h,
                            double2& c_double, CapyHiLo2& c_hilo)
{
    c_double = capy_map_pixel_double(cx, cy, stepX, stepY, px, py, w, h);
    c_hilo   = capy_map_pixel_hilo  (cx, cy, stepX, stepY, px, py, w, h);
}

// ----------------------------- Optional device logs ---------------------------
// Suggestion: call once per ~8k pixels or gated by gid%N==0 in your kernel.
#if defined(__CUDA_ARCH__)
CAPY_D void capy_log_map_init_if(uint32_t gid, int px, int py,
                                 double2 cD, const CapyHiLo2& cHL, int earlyIters)
{
#if CAPY_DEBUG_LOGGING
    if (!capy_should_log(gid, (uint32_t)CAPY_LOG_RATE)) return;
    char msg[256];
    // One ASCII line, deterministic; keep %.17e for double print parity.
    // Example:
    // CAPY map gid=123 px=10 py=20 cD=(1.23e-12,4.56e-12) cHL.hi=(...) lo=(...) early=64
    snprintf(msg, sizeof(msg),
             "CAPY map gid=%u px=%d py=%d cD=(%.17e,%.17e) cHL.hi=(%.17e,%.17e) lo=(%.17e,%.17e) early=%d",
             gid, px, py,
             cD.x, cD.y,
             cHL.x.hi, cHL.y.hi,
             cHL.x.lo, cHL.y.lo,
             earlyIters);
    LUCHS_LOG_DEVICE(msg);
#endif
}
#else
inline void capy_log_map_init_if(uint32_t, int, int, double2, const CapyHiLo2&, int) {}
#endif
