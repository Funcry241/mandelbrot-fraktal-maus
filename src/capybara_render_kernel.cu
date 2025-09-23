///// Otter: Mandelbrot render kernel using Capybara early Hi/Lo + classic continuation (fills d_iterations).
///// Schneefuchs: Header-only Capybara helpers; ASCII logs via LUCHS_LOG_HOST/DEVICE; one final device log per message.
///// Maus: Additive drop-in; keeps API simple (no PBO here). Launch via launch_mandelbrot_capybara(...).
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
    // Balanced config: good occupancy on SM80+ with modest register pressure.
    constexpr int BX = 32;
    constexpr int BY = 8;
    static_assert(BX > 0 && BY > 0, "Block dimensions must be positive");
}

// --------------------------------- helpers ------------------------------------
static __device__ __forceinline__ uint16_t clamp_u16_from_int(int v) {
    return (v < 0) ? 0u : (v > 65535 ? 65535u : static_cast<uint16_t>(v));
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

    // Compute iteration count with Capybara early phase + classic continuation
    const int iters = capy_compute_iters_from_zero(cx, cy, stepX, stepY, px, py, w, h, maxIter);
    d_it[py * w + px] = clamp_u16_from_int(iters);
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

    mandelbrotKernel_capybara<<<grid, block, 0, stream>>>(d_it, w, h, cx, cy, stepX, stepY, maxIter);
    CAPY_NT_CHECK(cudaPeekAtLastError());
}
