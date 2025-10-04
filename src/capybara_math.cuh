///// Otter: Capybara scaffold – Hi/Lo arithmetic, renorm, and device ASCII telemetry.
///// This file is additive and safe to include from device/host code.
///// Schneefuchs: Only one final LUCHS_LOG_DEVICE call per message; snprintf only for message construction.
///// Maus: No API break; rate-limited logs; compile-time defaults; ASCII-only.
///// Datei: src/capybara_math.cuh

#pragma once
#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define CAPY_HD __host__ __device__ __forceinline__
#define CAPY_D  __device__ __forceinline__
#else
#define CAPY_HD inline
#define CAPY_D  inline
#endif

// ----------------------------- Compile-time defaults --------------------------
// You can override these via compiler definitions or a central settings header later.
#ifndef CAPY_ENABLED
#define CAPY_ENABLED 1
#endif

#ifndef CAPY_EARLY_ITERS
#define CAPY_EARLY_ITERS 64
#endif

// 2^-48 ~= 3.5527136788e-15 – threshold when the low part is considered "too large" relative to hi
#ifndef CAPY_RENORM_RATIO
#define CAPY_RENORM_RATIO 3.5527136788005009e-15
#endif

#ifndef CAPY_MAPPING_EXACT_STEP
#define CAPY_MAPPING_EXACT_STEP 1
#endif

// Logging control for device telemetry; keep this 0 unless debugging
#ifndef CAPY_DEBUG_LOGGING
#define CAPY_DEBUG_LOGGING 0
#endif

// Log every N-th occurrence (power-of-two not required)
#ifndef CAPY_LOG_RATE
#define CAPY_LOG_RATE 8192
#endif

// ----------------------------- Telemetry (device) -----------------------------
#if defined(__CUDA_ARCH__)
#include "luchs_log_device.hpp"
#endif

// Rate-limit helper (works on both host and device)
CAPY_HD bool capy_should_log(uint32_t key, uint32_t rate)
{
    if (rate == 0u) return false;
    return (key % rate) == 0u;
}

#if defined(__CUDA_ARCH__)
CAPY_D void capy_log_init(uint32_t gid, int px, int py, int earlyIters)
{
#if CAPY_DEBUG_LOGGING
    if (!capy_should_log(gid, (uint32_t)CAPY_LOG_RATE)) return;
    char msg[192];
    // ASCII only, one line:
    // Example: CAPY init gid=12345 px=10 py=20 early=64
    snprintf(msg, sizeof(msg), "CAPY init gid=%u px=%d py=%d early=%d", gid, px, py, earlyIters);
    LUCHS_LOG_DEVICE(msg);
#endif
}

CAPY_D void capy_log_renorm(uint32_t gid, int iter, double hiAbs, double loAbs)
{
#if CAPY_DEBUG_LOGGING
    if (!capy_should_log(gid + (uint32_t)iter, (uint32_t)CAPY_LOG_RATE)) return;
    char msg[192];
    // Example: CAPY renorm gid=12345 it=17 hi=1.23e-12 lo=9.87e-15
    snprintf(msg, sizeof(msg), "CAPY renorm gid=%u it=%d hi=%.17e lo=%.17e", gid, iter, hiAbs, loAbs);
    LUCHS_LOG_DEVICE(msg);
#endif
}

CAPY_D void capy_log_step(uint32_t gid, int iter, double zx, double zy)
{
#if CAPY_DEBUG_LOGGING
    if (!capy_should_log(gid + (uint32_t)iter, (uint32_t)CAPY_LOG_RATE)) return;
    char msg[192];
    // Example: CAPY step gid=12345 it=8 zx=... zy=...
    snprintf(msg, sizeof(msg), "CAPY step gid=%u it=%d zx=%.17e zy=%.17e", gid, iter, zx, zy);
    LUCHS_LOG_DEVICE(msg);
#endif
}
#else
// Host no-op stubs keep call sites clean
inline void capy_log_init(uint32_t, int, int, int) {}
inline void capy_log_renorm(uint32_t, int, double, double) {}
inline void capy_log_step(uint32_t, int, double, double) {}
#endif

// ----------------------------- Hi/Lo primitives -------------------------------
struct CapyHiLo {
    double hi;
    double lo;
    CAPY_HD CapyHiLo() : hi(0.0), lo(0.0) {}
    CAPY_HD explicit CapyHiLo(double v) : hi(v), lo(0.0) {}
};

struct CapyHiLo2 {
    CapyHiLo x;
    CapyHiLo y;
    CAPY_HD CapyHiLo2() : x(), y() {}
    CAPY_HD CapyHiLo2(double xr, double yr) : x(xr), y(yr) {}
};

// Quick-Two-Sum: assumes |a| >= |b|
CAPY_HD CapyHiLo capy_quick_two_sum(double a, double b)
{
    CapyHiLo r;
    r.hi = a + b;
    r.lo = b - (r.hi - a);
    return r;
}

// Veltkamp Two-Sum: robust for any a, b
CAPY_HD CapyHiLo capy_two_sum(double a, double b)
{
    CapyHiLo r;
    r.hi = a + b;
    double z = r.hi - a;
    r.lo = (a - (r.hi - z)) + (b - z);
    return r;
}

// Add y into x (compensated)
CAPY_HD void capy_add(CapyHiLo& x, double y)
{
    CapyHiLo s = capy_two_sum(x.hi, y);
    s.lo += x.lo;
    // renormalize via quick two-sum (|s.hi| >= |s.lo| is usually true now)
    CapyHiLo t = capy_quick_two_sum(s.hi, s.lo);
    x.hi = t.hi;
    x.lo = t.lo;
}

// Add Hi/Lo into x
CAPY_HD void capy_add(CapyHiLo& x, const CapyHiLo& y)
{
    CapyHiLo s = capy_two_sum(x.hi, y.hi);
    s.lo += (x.lo + y.lo);
    CapyHiLo t = capy_quick_two_sum(s.hi, s.lo);
    x.hi = t.hi;
    x.lo = t.lo;
}

// Renormalize if low part grows too large relative to hi
CAPY_HD bool capy_renorm_if_needed(CapyHiLo& v)
{
    double ahi = fabs(v.hi);
    double alo = fabs(v.lo);
    if (alo > ahi * (double)CAPY_RENORM_RATIO) {
        // fold lo back into hi
        CapyHiLo t = capy_quick_two_sum(v.hi, v.lo);
        v.hi = t.hi;
        v.lo = t.lo;
        return true;
    }
    return false;
}

// ----------------------------- Mapping helpers --------------------------------
// Decompose scale into mantissa*2^e to keep steps nicely aligned at deep zooms.
CAPY_HD void capy_split_step(double step, double& mant, int& expo)
{
    // frexp: step = mant * 2^expo, mant in [0.5, 1)
    mant = frexp(step, &expo);
}

// Compute delta = off * step using the mantissa+exponent split to maintain structure
CAPY_HD double capy_scaled_delta(double off, double stepMant, int stepExpo)
{
#if CAPY_MAPPING_EXACT_STEP
    // multiply first in mantissa space, then scale by 2^expo
    double dm = off * stepMant;
    return ldexp(dm, stepExpo);
#else
    return off * (ldexp(stepMant, stepExpo));
#endif
}

// Build Hi/Lo coordinate: center (+) delta
CAPY_HD CapyHiLo capy_center_plus_delta(double center, double off, double step)
{
    double mant; int expo;
    capy_split_step(step, mant, expo);
    double delta = capy_scaled_delta(off, mant, expo);
    // Compensated center + delta
    CapyHiLo s = capy_two_sum(center, delta);
    // Keep as-is; renorm happens dynamics-aware later
    return s;
}

// Convenience for (x,y)
CAPY_HD CapyHiLo2 capy_map_center_step(double cx, double cy,
                                       double stepX, double stepY,
                                       double offx, double offy)
{
    CapyHiLo2 out;
    out.x = capy_center_plus_delta(cx, offx, stepX);
    out.y = capy_center_plus_delta(cy, offy, stepY);
    return out;
}

