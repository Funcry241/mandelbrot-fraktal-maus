///// Otter: Nacktmull – Mandelbrot kernel split (Classic vs. Deep) with host-side mode gating; no logic change, just specialization
///// Schneefuchs: API unverändert; ASCII-Logs; optional CUDA-event timing; inclusive-iter semantics; no fast-math flags
///// Maus: Block 32x8; exact cardioid/bulb; deterministic; SM80–SM90 sweetspot; per-frame gating, zero per-thread mode branches
///// Datei: src/capybara_render_kernel.cu
#include "pch.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "capybara_api.cuh"

// Capybara building blocks
#include "capybara_math.cuh"
#include "capybara_mapping.cuh"
#include "capybara_ziter.cuh"
#include "capybara_integration.cuh"
#include "capybara_pixel_iter.cuh"

// ------------------------------ launch config ---------------------------------
namespace {
    // Aus Settings gespiegelt (MANDEL_BLOCK_X/Y)
    constexpr int BX = Settings::MANDEL_BLOCK_X;
    constexpr int BY = Settings::MANDEL_BLOCK_Y;
    static_assert(BX > 0 && BY > 0, "Block dimensions must be positive");

    // Threshold für Deep-Path (host-seitig entschieden, keine per-thread Branches)
    constexpr double kBaseStepThresh = 8e-13;

    static inline double dyn_step_thresh_host(int maxIter) {
        if (maxIter <= 1024)  return kBaseStepThresh * 2.0;
        if (maxIter <= 4096)  return kBaseStepThresh;
        if (maxIter <= 16384) return kBaseStepThresh * 0.75;
        return kBaseStepThresh * 0.5;
    }
}

// --------------------------------- helpers ------------------------------------
static __device__ __forceinline__ uint16_t clamp_u16_from_int(int v) {
    return (v < 0) ? 0u : (v > 65535 ? 65535u : static_cast<uint16_t>(v));
}

// Analytic interior tests (exact): main cardioid and period-2 bulb
static __device__ __forceinline__ bool in_main_cardioid(double2 c) {
    const double x  = c.x - 0.25;
    const double y  = c.y;
    const double y2 = y * y;
    const double q  = x * x + y2;
    // Inside if q * (q + x) <= 0.25 * y^2
    return q * (q + x) <= 0.25 * y2;
}
static __device__ __forceinline__ bool in_period2_bulb(double2 c) {
    const double xr = c.x + 1.0;
    const double yr = c.y;
    // Inside if (x+1)^2 + y^2 <= (1/4)^2
    return (xr * xr + yr * yr) <= (1.0 / 16.0);
}
static __device__ __forceinline__ bool in_cardioid_or_bulb(double2 c) {
    // Bulb-Test zuerst: gleicher Wahrheitswert, minimal günstiger im häufigen Outside-Fall.
    return in_period2_bulb(c) || in_main_cardioid(c);
}

// ------------------------------ classic kernel --------------------------------
__global__ __launch_bounds__(BX * BY, 2) // ggf. 3 testen, wenn Reg-Budget es zulässt
void mandelbrotKernel_classic(
    uint16_t* __restrict__ d_it,
    int w, int h,
    double cx, double cy,
    double stepX, double stepY,
    int maxIter)
{
    const int px  = blockIdx.x * blockDim.x + threadIdx.x;
    const int py  = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    const int idx = py * w + px;

    // Map pixel -> complex plane (double). FMA-Form, mit gecachten Halbmaßen.
    const double halfW = 0.5 * static_cast<double>(w);
    const double halfH = 0.5 * static_cast<double>(h);
    const double pxD   = static_cast<double>(px);
    const double pyD   = static_cast<double>(py);
    const double x     = fma(pxD - halfW, stepX, cx);
    const double y     = fma(pyD - halfH, stepY, cy);
    const double2 cD   = make_double2(x, y);

    // 1) Analytic interior: exact membership -> it = maxIter
    if (in_cardioid_or_bulb(cD)) {
        d_it[idx] = clamp_u16_from_int(maxIter);
        return;
    }

    // 2) Classic escape-time, inclusive semantics
    double zx = 0.0, zy = 0.0;
    int it = 0;
    #pragma unroll 1
    for (; it < maxIter; ++it) {
        const double xx = zx * zx - zy * zy + cD.x;
        const double yy = fma(2.0 * zx, zy, cD.y); // 2*zx*zy + cD.y
        zx = xx; zy = yy;
        const double r2 = fma(xx, xx, yy * yy);    // xx*xx + yy*yy
        if (r2 > 4.0) { ++it; break; }
    }
    d_it[idx] = clamp_u16_from_int(it);
}

// ------------------------------- deep kernel ----------------------------------
__global__ __launch_bounds__(BX * BY, 2) // ggf. 3 testen, wenn Reg-Budget es zulässt
void mandelbrotKernel_capybara_deep(
    uint16_t* __restrict__ d_it,
    int w, int h,
    double cx, double cy,
    double stepX, double stepY,
    int maxIter)
{
    const int px  = blockIdx.x * blockDim.x + threadIdx.x;
    const int py  = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    const int idx = py * w + px;

    // FMA-Form für Mapping (identische Numerik), mit gecachten Halbmaßen.
    const double halfW = 0.5 * static_cast<double>(w);
    const double halfH = 0.5 * static_cast<double>(h);
    const double pxD   = static_cast<double>(px);
    const double pyD   = static_cast<double>(py);
    const double x     = fma(pxD - halfW, stepX, cx);
    const double y     = fma(pyD - halfH, stepY, cy);
    const double2 cD   = make_double2(x, y);

    if (in_cardioid_or_bulb(cD)) {
        d_it[idx] = clamp_u16_from_int(maxIter);
        return;
    }

    const int iters = capy_compute_iters_from_zero(cx, cy, stepX, stepY, px, py, w, h, maxIter);
    d_it[idx] = clamp_u16_from_int(iters);
}

// ------------------------------- host wrapper ---------------------------------
#define CAPY_NT_CHECK(call) \
    do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
        LUCHS_LOG_HOST("[CUDA][CAPY] rc=%d at %s:%d", (int)_e, __FILE__, __LINE__); } } while (0)

extern "C" void launch_mandelbrot_capybara(
    uint16_t* d_it,
    int w, int h,
    double cx, double cy,
    double stepX, double stepY,
    int maxIter,
    cudaStream_t stream /*= nullptr*/)
{
    if (!d_it || w <= 0 || h <= 0 || maxIter < 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CAPY][NACKTMULL] invalid-args w=%d h=%d maxIter=%d d_it=%p", w, h, maxIter, (void*)d_it);
        }
        return;
    }

    const dim3 block(BX, BY);
    const dim3 grid((w + BX - 1) / BX, (h + BY - 1) / BY);

    const double ax = fabs(stepX);
    const double ay = fabs(stepY);
    const double m  = (ax > ay ? ax : ay);
    const double kThresh = dyn_step_thresh_host(maxIter);
    const bool useClassic = (m > kThresh);

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        LUCHS_LOG_HOST("[CAPY][NACKTMULL] queued w=%d h=%d grid=%dx%d block=%dx%d maxIter=%d mode=%s stream=%p",
                       w, h, grid.x, grid.y, block.x, block.y, maxIter,
                       useClassic ? "classic" : "deep", (void*)stream);
    }

    cudaEvent_t evStart = nullptr, evStop = nullptr;
    if constexpr (Settings::performanceLogging) {
        (void)cudaEventCreateWithFlags(&evStart, cudaEventDefault);
        (void)cudaEventCreateWithFlags(&evStop,  cudaEventDefault);
        (void)cudaEventRecord(evStart, stream);
    }

    if (useClassic) {
        mandelbrotKernel_classic<<<grid, block, 0, stream>>>(d_it, w, h, cx, cy, stepX, stepY, maxIter);
    } else {
        mandelbrotKernel_capybara_deep<<<grid, block, 0, stream>>>(d_it, w, h, cx, cy, stepX, stepY, maxIter);
    }

    if constexpr (Settings::performanceLogging) {
        (void)cudaEventRecord(evStop, stream);
        (void)cudaEventSynchronize(evStop);
        float ms = 0.0f;
        (void)cudaEventElapsedTime(&ms, evStart, evStop);
        LUCHS_LOG_HOST("[CAPY][NACKTMULL][time] mand=%.3f ms (w=%d h=%d it=%d)", (double)ms, w, h, maxIter);
        (void)cudaEventDestroy(evStart);
        (void)cudaEventDestroy(evStop);
    }

    CAPY_NT_CHECK(cudaPeekAtLastError());
}
