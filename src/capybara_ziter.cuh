///// Otter: Early-Iter (Hi/Lo) z = z^2 + c with renorm + ASCII telemetry (Capybara).
///// Schneefuchs: Header-only; device-inline; one final LUCHS_LOG_DEVICE per message; snprintf only for construction.
///// Maus: Bridge-friendly: returns iters-done; fold-to-double helper for seamless handoff.
///// Datei: src/capybara_ziter.cuh

#pragma once
#include <math.h>
#include <stdint.h>
#include "capybara_math.cuh"   // CapyHiLo, CapyHiLo2, two-sum, renorm, logs, CAPY_* macros

#if defined(__CUDACC__)
#define CAPY_HD __host__ __device__ __forceinline__
#define CAPY_D  __device__ __forceinline__
#else
#define CAPY_HD inline
#define CAPY_D  inline
#endif

// ----------------------------- Helpers: fold & convert ------------------------
CAPY_HD void capy_fold(CapyHiLo& v)
{
    // Fold lo back into hi (exact quick-two-sum)
    CapyHiLo t = capy_quick_two_sum(v.hi, v.lo);
    v.hi = t.hi;
    v.lo = t.lo; // should be tiny post-fold
}

CAPY_HD void capy_fold(CapyHiLo2& v2)
{
    capy_fold(v2.x);
    capy_fold(v2.y);
}

CAPY_HD double2 capy_to_double2_folded(const CapyHiLo2& v2)
{
    // Prefer returning the (hi+lo) sum to preserve accuracy when handing off to classic double path
    CapyHiLo sx = capy_quick_two_sum(v2.x.hi, v2.x.lo);
    CapyHiLo sy = capy_quick_two_sum(v2.y.hi, v2.y.lo);
    double2 r; r.x = sx.hi; r.y = sy.hi;
    return r;
}

// Norm^2 using hi components (conservative bailout in early tiny-z regime)
CAPY_HD double capy_norm2_hi(const CapyHiLo2& z)
{
    return z.x.hi * z.x.hi + z.y.hi * z.y.hi;
}

// ----------------------------- Core: z^2 + c (Hi/Lo) -------------------------
// Computes z = z^2 + c using hi for the main term and FMA-backed residuals to
// absorb first-order error from the lo parts. Keeps cost low for early-phase.
CAPY_HD void capy_square_add(CapyHiLo2& z, const CapyHiLo2& c)
{
    // Aliases for readability
    const double xh = z.x.hi, xl = z.x.lo;
    const double yh = z.y.hi, yl = z.y.lo;

    // xx ≈ (xh+xl)^2 = xh^2 + 2*xh*xl + O(xl^2)
    double xx = xh * xh;
#if defined(__CUDA_ARCH__) || defined(__cpp_lib_fma) || (__cplusplus >= 201103L)
    double exx = fma(xh, xh, -xx); // exact residual of xh^2 if FMA
#else
    double exx = 0.0;
#endif
    exx += 2.0 * xh * xl; // first-order correction

    // yy ≈ (yh+yl)^2
    double yy = yh * yh;
#if defined(__CUDA_ARCH__) || defined(__cpp_lib_fma) || (__cplusplus >= 201103L)
    double eyy = fma(yh, yh, -yy);
#else
    double eyy = 0.0;
#endif
    eyy += 2.0 * yh * yl;

    // xy ≈ (xh+xl)*(yh+yl) = xh*yh + xh*yl + yh*xl + O(xl*yl)
    double xy = xh * yh;
#if defined(__CUDA_ARCH__) || defined(__cpp_lib_fma) || (__cplusplus >= 201103L)
    double exy = fma(xh, yh, -xy);
#else
    double exy = 0.0;
#endif
    exy += xh * yl + yh * xl;

    // Compute new real/imag using compensated adds
    // rx = (xx - yy) + c.x
    {
        double main = (xx - yy);
#if defined(__CUDA_ARCH__) || defined(__cpp_lib_fma) || (__cplusplus >= 201103L)
        double err  = (exx - eyy);
#else
        double err  = (exx - eyy);
#endif
        // hi add with c.x.hi
        CapyHiLo s = capy_two_sum(main, c.x.hi);
        s.lo += err + c.x.lo;
        CapyHiLo t = capy_quick_two_sum(s.hi, s.lo);
        z.x.hi = t.hi;
        z.x.lo = t.lo;
    }

    // ry = 2*xy + c.y
    {
        double main = 2.0 * xy;
#if defined(__CUDA_ARCH__) || defined(__cpp_lib_fma) || (__cplusplus >= 201103L)
        double err  = 2.0 * exy;
#else
        double err  = 2.0 * exy;
#endif
        CapyHiLo s = capy_two_sum(main, c.y.hi);
        s.lo += err + c.y.lo;
        CapyHiLo t = capy_quick_two_sum(s.hi, s.lo);
        z.y.hi = t.hi;
        z.y.lo = t.lo;
    }
}

// --------------------- Early-Iter Loop with Renorm & Telemetry ----------------
CAPY_D int capy_early_iterate(CapyHiLo2& z, const CapyHiLo2& c,
                              int earlyIters, uint32_t gid)
{
#if CAPY_ENABLED
    if (earlyIters <= 0) return 0;
    int it = 0;
    for (; it < earlyIters; ++it) {
        capy_square_add(z, c);

        // Optional per-step telemetry (rate-limited)
        capy_log_step(gid, it, z.x.hi, z.y.hi);

        // Bailout test on hi parts (adequate in early regime)
        double n2 = capy_norm2_hi(z);
        if (n2 > 4.0) { ++it; break; } // count the step that escaped

        // Renormalize if low parts grew comparatively large
        bool r1 = capy_renorm_if_needed(z.x);
        bool r2 = capy_renorm_if_needed(z.y);
        if (r1 || r2) {
            capy_log_renorm(gid, it, fabs(z.x.hi) + fabs(z.y.hi),
                                   fabs(z.x.lo) + fabs(z.y.lo));
        }
    }
    return it;
#else
    (void)z; (void)c; (void)earlyIters; (void)gid;
    return 0;
#endif
}

// --------------------- Zero init + bridge convenience helpers -----------------
CAPY_HD CapyHiLo2 capy_zero_hilo2()
{
    return CapyHiLo2(0.0, 0.0);
}

// Run early iterations from z=0 with Hi/Lo, then hand off as double2.
// Returns the number of iterations performed in early phase.
CAPY_D int capy_early_from_zero(const CapyHiLo2& cHL, int earlyIters,
                                uint32_t gid, double2& z_out_double)
{
    CapyHiLo2 z = capy_zero_hilo2();
    capy_log_init(gid, /*px*/0, /*py*/0, earlyIters); // px/py unknown here; caller may prefer capy_log_map_init_if
    int done = capy_early_iterate(z, cHL, earlyIters, gid);
    // Fold and convert for downstream classic path
    z_out_double = capy_to_double2_folded(z);
    return done;
}

// Variant: continue from an existing double2 z-in, but promote to Hi/Lo first.
// Useful if you already did a few classic iterations before enabling Capybara.
CAPY_D int capy_early_from_double(double2 z_in, const CapyHiLo2& cHL, int earlyIters,
                                  uint32_t gid, double2& z_out_double)
{
    CapyHiLo2 z;
    z.x = CapyHiLo(z_in.x);
    z.y = CapyHiLo(z_in.y);
    capy_log_init(gid, /*px*/0, /*py*/0, earlyIters);
    int done = capy_early_iterate(z, cHL, earlyIters, gid);
    z_out_double = capy_to_double2_folded(z);
    return done;
}
