///// Otter: Capybara pixel-iter — drop-in device helpers to compute Mandelbrot iterations with early Hi/Lo + classic continuation.
///// Schneefuchs: Header-only; device-inline; ASCII one-liners via capy_*; no API break; defaults configurable via CAPY_* macros.
///// Maus: Returns either just the iteration count or (iters, final z). Escape radius^2 kept at 4.0 for parity.
///// Datei: src/capybara_pixel_iter.cuh

#pragma once
#include <math.h>
#include <stdint.h>

#include "capybara_math.cuh"        // Hi/Lo primitives + telemetry
#include "capybara_mapping.cuh"     // pixel→complex (double & Hi/Lo)
#include "capybara_ziter.cuh"       // early Hi/Lo iteration
#include "capybara_integration.cuh" // bridge helpers

#if defined(__CUDACC__)
#define CAPY_HD __host__ __device__ __forceinline__
#define CAPY_D  __device__ __forceinline__
#else
#define CAPY_HD inline
#define CAPY_D  inline
#endif

// ----------------------------- Core compute (z=0) -----------------------------
// Computes Mandelbrot iteration count for pixel (px,py) with center (cx,cy), steps (stepX,stepY).
// Starts from z = 0, performs Capybara early iterations (if enabled) and continues classically.
// Returns the number of iterations taken until escape or maxIter (inclusive of the last step).
CAPY_D int capy_compute_iters_from_zero(double cx, double cy,
                                        double stepX, double stepY,
                                        int px, int py, int w, int h,
                                        int maxIter)
{
    // Prepare mapping + gid + optional rate-limited init log
    double2 cD; CapyHiLo2 cHL; uint32_t gid;
    capy_prepare_c(cx, cy, stepX, stepY, px, py, w, h, cD, cHL, gid);

    // Early Hi/Lo segment
    double2 z = make_double2(0.0, 0.0);
    int it = capy_mandelbrot_early(z, cD, cHL, maxIter, gid);
    if (it >= maxIter) return it;

    // If we already escaped in early phase, the classic loop will bail immediately
    // Keep the classic continuation compact and branchless where possible
    for (; it < maxIter; ++it) {
        const double x = z.x, y = z.y;
        const double xx = x * x - y * y + cD.x;
        const double yy = 2.0 * x * y + cD.y;
        z.x = xx; z.y = yy;
        if (xx * xx + yy * yy > 4.0) { ++it; break; }
    }
    return it;
}

// ----------------------------- Core compute (z!=0) ----------------------------
// Variant that starts from an existing z0 (double2). Useful for tiling or progressive paths.
// Returns iterations added on top of the caller-provided starting iteration 'it0'.
// The caller is expected to add the return value to its own iteration counter if needed.
CAPY_D int capy_compute_iters_from_z(double cx, double cy,
                                     double stepX, double stepY,
                                     int px, int py, int w, int h,
                                     int it0, int maxIter,
                                     /*inout*/ double2& z /* will be advanced */)
{
    // Prepare mapping + gid + optional rate-limited init log
    double2 cD; CapyHiLo2 cHL; uint32_t gid;
    capy_prepare_c(cx, cy, stepX, stepY, px, py, w, h, cD, cHL, gid);

    // Early Hi/Lo segment continuing from z
    const int budget = (maxIter - it0) > 0 ? (maxIter - it0) : 0;
    int done = 0;
    if (budget > 0) {
        done = capy_early_from_z(z, z, cHL, gid, budget);
        if (done >= budget) return done; // reached maxIter within early phase
    }

    // Classic continuation for the remainder
    int i = 0;
    for (; i + done < budget; ++i) {
        const double x = z.x, y = z.y;
        const double xx = x * x - y * y + cD.x;
        const double yy = 2.0 * x * y + cD.y;
        z.x = xx; z.y = yy;
        if (xx * xx + yy * yy > 4.0) { ++i; break; }
    }
    return done + i;
}

// ----------------------------- Convenience (with z) ---------------------------
// Computes both iterations and returns the final z via out parameter (z_out).
// Starts from z=0; useful if the kernel wants to keep z for coloring.
CAPY_D int capy_compute_iters_and_z(double cx, double cy,
                                    double stepX, double stepY,
                                    int px, int py, int w, int h,
                                    int maxIter,
                                    /*out*/ double2& z_out)
{
    // Prepare mapping + gid + optional rate-limited init log
    double2 cD; CapyHiLo2 cHL; uint32_t gid;
    capy_prepare_c(cx, cy, stepX, stepY, px, py, w, h, cD, cHL, gid);

    // Early Hi/Lo segment
    double2 z = make_double2(0.0, 0.0);
    int it = capy_mandelbrot_early(z, cD, cHL, maxIter, gid);
    if (it >= maxIter) { z_out = z; return it; }

    // Classic continuation
    for (; it < maxIter; ++it) {
        const double x = z.x, y = z.y;
        const double xx = x * x - y * y + cD.x;
        const double yy = 2.0 * x * y + cD.y;
        z.x = xx; z.y = yy;
        if (xx * xx + yy * yy > 4.0) { ++it; break; }
    }
    z_out = z;
    return it;
}
