///// Otter: Capybara bridge helpers to plug Hi/Lo early-iter into existing Mandelbrot kernels (no API break).
///// Schneefuchs: Header-only; device-inline; ASCII logs rate-limited via capy_*; zero host/device sync changes.
///// Maus: Provides prepare_c(), early_from_zero/early_from_z(), and a one-call capy_mandelbrot_early().
///// Datei: src/capybara_integration.cuh

#pragma once
#include <math.h>
#include <stdint.h>

#include "capybara_math.cuh"      // Hi/Lo primitives + telemetry
#include "capybara_mapping.cuh"   // pixel→complex (double & Hi/Lo)
#include "capybara_ziter.cuh"     // early Hi/Lo iteration

#if defined(__CUDACC__)
#define CAPY_HD __host__ __device__ __forceinline__
#define CAPY_D  __device__ __forceinline__
#else
#define CAPY_HD inline
#define CAPY_D  inline
#endif

// --------------------------------- Config hooks --------------------------------
// If you later expose settings in settings.hpp, you can #define these macros there.
#ifndef CAPY_EFFECTIVE_ENABLED
#define CAPY_EFFECTIVE_ENABLED() (CAPY_ENABLED)
#endif

#ifndef CAPY_EFFECTIVE_EARLY_ITERS
#define CAPY_EFFECTIVE_EARLY_ITERS() (CAPY_EARLY_ITERS)
#endif

// --------------------------------- Prepare c ----------------------------------
// Computes both classic double2 'cD' and Hi/Lo 'cHL' + a deterministic gid.
// Optional: emits a single rate-limited mapping log when CAPY_DEBUG_LOGGING=1.
CAPY_D void capy_prepare_c(double cx, double cy,
                           double stepX, double stepY,
                           int px, int py, int w, int h,
                           /*out*/ double2& cD,
                           /*out*/ CapyHiLo2& cHL,
                           /*out*/ uint32_t& gid)
{
    gid = capy_gid(px, py, w);
    capy_map_pixel(cx, cy, stepX, stepY, px, py, w, h, cD, cHL);
    capy_log_map_init_if(gid, px, py, cD, cHL, CAPY_EFFECTIVE_EARLY_ITERS());
}

// ------------------------------- Early from z=0 -------------------------------
// Runs early Hi/Lo iterations starting from z=0; returns iterations done and a folded double2 z.
CAPY_D int capy_early_from_zero_z(double2& z_out, const CapyHiLo2& cHL, uint32_t gid, int budget)
{
    if (!CAPY_EFFECTIVE_ENABLED() || budget <= 0) { z_out = make_double2(0.0, 0.0); return 0; }
    return capy_early_from_zero(cHL, budget, gid, z_out);
}

// ------------------------------ Early from z!=0 -------------------------------
// Promotes incoming double2 z to Hi/Lo, runs up to 'budget' early iterations, returns iters done.
CAPY_D int capy_early_from_z(double2 z_in, double2& z_out, const CapyHiLo2& cHL, uint32_t gid, int budget)
{
    if (!CAPY_EFFECTIVE_ENABLED() || budget <= 0) { z_out = z_in; return 0; }
    return capy_early_from_double(z_in, cHL, budget, gid, z_out);
}

// ------------------------- One-call kernel-side helper ------------------------
// Plug this into your Mandelbrot kernel near the top, after computing 'cD' and 'gid'.
// Usage pattern:
//
//   double2 cD; CapyHiLo2 cHL; uint32_t gid;
//   capy_prepare_c(cx, cy, stepX, stepY, px, py, w, h, cD, cHL, gid);
//   int it = 0; double2 z = make_double2(0.0, 0.0);
//   it += capy_mandelbrot_early(z, cD, cHL, maxIter, gid);
//   // continue with your classic double path for the remaining (maxIter - it) iterations.
//
CAPY_D int capy_mandelbrot_early(double2& z, const double2& cD, const CapyHiLo2& cHL,
                                 int maxIter, uint32_t gid)
{
    if (!CAPY_EFFECTIVE_ENABLED()) return 0;
    int budget = CAPY_EFFECTIVE_EARLY_ITERS();
    if (budget <= 0) return 0;
    if (budget > maxIter) budget = maxIter;

    // If z is (0,0) we can use the faster from-zero variant; cheap test:
    const bool z_is_zero = (z.x == 0.0) & (z.y == 0.0);
    int done = 0;
    if (z_is_zero) {
        done = capy_early_from_zero_z(z, cHL, gid, budget);
    } else {
        done = capy_early_from_z(z, z, cHL, gid, budget);
    }

    // Edge case: if we escaped during early phase, z now holds the last folded value already.
    // The classic path should detect norm^2>4 immediately and stop.
    (void)cD; // kept for symmetry / future checks; not used here.
    return done;
}
