///// Otter: Mandelbrot render kernel using Capybara early Hi/Lo + classic continuation (fills d_iterations). Sweetspot: 32x16 blocks on SM80–SM90, dynamic Hi/Lo gating via kBaseStepThresh=8e-13 (maxIter-aware).
///// Schneefuchs: API unverändert; ASCII-Logs; optionale CUDA-Event-Zeitmessung bei Settings::performanceLogging; identische Iterationszahlen, nur schnellerer Pfadwechsel.
// ///// Maus: Zero information loss – Innenpunkte = maxIter; Hi/Lo nur bei feinem Pixelstep; Host/Device sauber getrennt. Deterministic logs, no fast-math switches.
// ///// Datei: src/capybara_render_kernel.cu

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
    // Sweetspot: 32x16 keeps 512 threads/block, good latency hiding on SM80+ given typical register pressure.
    constexpr int BX = 32;
    constexpr int BY = 16;
    static_assert(BX > 0 && BY > 0, "Block dimensions must be positive");

    // Base threshold for switching to Capybara Hi/Lo when pixel steps get very fine.
    // Tuned for SM80–SM90. We modulate by maxIter to enter Hi/Lo earlier for large iteration budgets.
    constexpr double kBaseStepThresh = 8e-13;

    __device__ __forceinline__ double dyn_step_thresh(int maxIter) {
        // Conservative, step-wise schedule keeps results identical while improving perf across common maxIter.
        // Larger maxIter → engage Hi/Lo earlier (smaller threshold).
        if (maxIter <= 1024)  return kBaseStepThresh * 2.0;   // shallow budgets → classic path longer
        if (maxIter <= 4096)  return kBaseStepThresh;         // default
        if (maxIter <= 16384) return kBaseStepThresh * 0.75;  // deeper budgets
        return kBaseStepThresh * 0.5;                         // very deep budgets
    }
}

// --------------------------------- helpers ------------------------------------
static __device__ __forceinline__ uint16_t clamp_u16_from_int(int v) {
    return (v < 0) ? 0u : (v > 65535 ? 65535u : static_cast<uint16_t>(v));
}

// Analytic interior tests (exact): main cardioid and period-2 bulb
static __device__ __forceinline__ bool in_main_cardioid(double2 c) {
    const double x = c.x - 0.25;
    const double y = c.y;
    const double q = x * x + y * y;
    // Inside if q * (q + x) <= 0.25 * y^2
    return q * (q + x) <= 0.25 * (y * y);
}
static __device__ __forceinline__ bool in_period2_bulb(double2 c) {
    const double xr = c.x + 1.0;
    const double yr = c.y;
    // Inside if (x+1)^2 + y^2 <= (1/4)^2
    return (xr * xr + yr * yr) <= (1.0 / 16.0);
}
static __device__ __forceinline__ bool in_cardioid_or_bulb(double2 c) {
    return in_main_cardioid(c) || in_period2_bulb(c);
}

// -------------------------------- render kernel --------------------------------
// Computes iteration counts only. Coloring/heatmap happens elsewhere.
__global__ __launch_bounds__(BX * BY, 2)
void mandelbrotKernel_capybara(
    uint16_t* __restrict__ d_it,
    int w, int h,
    double cx, double cy,
    double stepX, double stepY,
    int maxIter)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    const int idx = py * w + px;

    // Map pixel -> complex plane (double). Keep it branch-free and deterministic.
    const double x = cx + (static_cast<double>(px) - 0.5 * static_cast<double>(w)) * stepX;
    const double y = cy + (static_cast<double>(py) - 0.5 * static_cast<double>(h)) * stepY;
    const double2 cD = make_double2(x, y);

    // 1) Analytic interior: exact membership → it = maxIter (no iterations needed)
    if (in_cardioid_or_bulb(cD)) {
        d_it[idx] = clamp_u16_from_int(maxIter);
        return;
    }

    // 2) Hi/Lo gating: for coarse pixel steps use classic double escape-time (identical result)
    const double ax = fabs(stepX);
    const double ay = fabs(stepY);
    const double m  = (ax > ay ? ax : ay);
    const double kThresh = dyn_step_thresh(maxIter);
    if (m > kThresh) {
        // Classic double precision path; check AFTER update (inclusive count), matching previous semantics.
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
        return;
    }

    // 3) Deep zoom path: Capybara early Hi/Lo + classic continuation (identical iteration counts)
    const int iters = capy_compute_iters_from_zero(cx, cy, stepX, stepY, px, py, w, h, maxIter);
    d_it[idx] = clamp_u16_from_int(iters);

    // single-line PTX "no-op": self-move on a dummy register (ptxas-safe across PTX versions)
    unsigned __ptx_dummy = 0u;
    asm volatile ("mov.u32 %0, %0;" : "+r"(__ptx_dummy));
}

// ------------------------------- host wrapper ---------------------------------
// Non-throwing numeric RC logs (avoid C4297 under /WX)
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
            LUCHS_LOG_HOST("[CAPY] invalid-args w=%d h=%d maxIter=%d d_it=%p", w, h, maxIter, (void*)d_it);
        }
        return;
    }

    const dim3 block(BX, BY);
    const dim3 grid((w + BX - 1) / BX, (h + BY - 1) / BY);

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        LUCHS_LOG_HOST("[CAPY] queued w=%d h=%d grid=%dx%d block=%dx%d maxIter=%d stream=%p",
                       w, h, grid.x, grid.y, block.x, block.y, maxIter, (void*)stream);
    }

    // Optional CUDA event timing (visible when Settings::performanceLogging == true)
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    if constexpr (Settings::performanceLogging) {
        (void)cudaEventCreateWithFlags(&evStart, cudaEventDefault);
        (void)cudaEventCreateWithFlags(&evStop,  cudaEventDefault);
        (void)cudaEventRecord(evStart, stream);
    }

    mandelbrotKernel_capybara<<<grid, block, 0, stream>>>(d_it, w, h, cx, cy, stepX, stepY, maxIter);

    if constexpr (Settings::performanceLogging) {
        (void)cudaEventRecord(evStop, stream);
        (void)cudaEventSynchronize(evStop);
        float ms = 0.0f;
        (void)cudaEventElapsedTime(&ms, evStart, evStop);
        LUCHS_LOG_HOST("[CAPY][time] mand=%.3f ms (w=%d h=%d it=%d)", (double)ms, w, h, maxIter);
        (void)cudaEventDestroy(evStart);
        (void)cudaEventDestroy(evStop);
    }

    CAPY_NT_CHECK(cudaPeekAtLastError());
}
