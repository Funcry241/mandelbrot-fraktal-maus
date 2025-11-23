///// Otter: Nacktmull - single-path classic continuation with warp-exit (no fast-math, no API change)
///// Schneefuchs: Header-only; device-inline; one runtime path; ASCII-only; inclusive-iter semantics preserved
///// Maus: Escape radius^2 = 4.0; returns iters; early Hi/Lo unchanged; no redundant fallbacks
///// Datei: src/capybara_pixel_iter.cuh

#pragma once
#include <math.h>
#include <stdint.h>

#include "capybara_math.cuh"        // Hi/Lo primitives + telemetry
#include "capybara_mapping.cuh"     // pixel->complex (double & Hi/Lo)
#include "capybara_ziter.cuh"       // early Hi/Lo iteration
#include "capybara_integration.cuh" // bridge helpers

#if defined(__CUDACC__)
#define CAPY_HD __host__ __device__ __forceinline__
#define CAPY_D  __device__ __forceinline__
#else
#define CAPY_HD inline
#define CAPY_D  inline
#endif

// ===== Nacktmull: single-path classic continuation with warp-exit =============
// Schneefuchs: one loop for all threads; a warp leaves together when none remain active.
// Note: Device-only body; host compiles remain parsable without providing an alternative runtime path.
CAPY_D int capy_classic_continue(double2& z, const double2 cD, int it, const int maxIter)
{
#if defined(__CUDA_ARCH__)
    const unsigned mask = __activemask();
    bool active = (it < maxIter);

    for (;;)
    {
        if (active)
        {
            const double x = z.x, y = z.y;
            const double xx = x * x - y * y + cD.x;
            const double yy = 2.0 * x * y + cD.y;
            z.x = xx; z.y = yy;

            // Inclusive iteration accounting (parity with original).
            ++it;

            // Escape / limit after performing this step.
            const double r2 = xx * xx + yy * yy;
            if (r2 > 4.0 || it >= maxIter)
                active = false;
        }

        // Exit when no thread in the warp remains active.
        const unsigned anyActive = __ballot_sync(mask, active);
        if (anyActive == 0u) break;
    }
    return it;
#else
    // Host-only translation units should not rely on this device helper.
    // Intentionally no alternative algorithm here to keep a single runtime path.
    return it;
#endif
}

// ------------------------ Core compute (prepared c / HiLo) ---------------------
// Uses precomputed cD / cHL / gid (capy_prepare_c already done by the caller).
// Starts from z = 0 and performs Capybara early iterations plus classic continuation.
// Returns the number of iterations taken until escape or maxIter (inclusive).
CAPY_D int capy_compute_iters_from_prepared(const double2& cD,
                                            const CapyHiLo2& cHL,
                                            uint32_t gid,
                                            int maxIter)
{
    double2 z = make_double2(0.0, 0.0);

    // Early Hi/Lo segment
    int it = capy_mandelbrot_early(z, cD, cHL, maxIter, gid);
    if (it >= maxIter) return it;

    // Nacktmull: single-path classic continuation with warp-exit
    it = capy_classic_continue(z, cD, it, maxIter);
    return it;
}

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

    // Delegate to the prepared-path helper to avoid duplicate mapping in callers.
    return capy_compute_iters_from_prepared(cD, cHL, gid, maxIter);
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

    // Nacktmull: continue via single-path helper
    const int it_before = it0 + done;
    const int it_after  = capy_classic_continue(z, cD, it_before, maxIter);
    return (it_after - it0); // iterations added on top of it0
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

    // Nacktmull: single-path classic continuation with warp-exit
    it = capy_classic_continue(z, cD, it, maxIter);
    z_out = z;
    return it;
}
