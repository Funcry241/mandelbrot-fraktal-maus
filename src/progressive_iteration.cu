///// Otter: Progressive Iteration & Resume (impl) + optional local Nacktmull-Settings (default off)
///// Schneefuchs: Messbare Pfade; ASCII-Logs; kein versteckter Funktionswechsel.
///// Maus: Coalesced SoA, fixed block, chunked inner loop.
///// Datei: src/progressive_iteration.cu

#include "progressive_iteration.cuh"
#include "luchs_log_host.hpp"
#include "luchs_log_device.hpp"
#include "settings.hpp"              // Host toggles (performanceLogging/debugLogging)

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

// -----------------------------------------------------------------------------
// Local, headerless Nacktmull settings (functional, but default OFF)
// This replaces the removed settings_nacktmull.hpp for this TU only.
// -----------------------------------------------------------------------------
namespace NacktmullSettings {
struct ProgressivePolicy {
    bool     enabled;                   // false → neutral (no influence)
    uint32_t sliceMin;                  // min addIter per step when enabled (>=1)
    double   sliceMaxPct;               // clamp addIter to <= sliceMaxPct * maxIterCap
    bool     deviceDebugLog;            // extra device logs from kernel
    double   stopThresholdSurvivorsPct; // suggest halt if survivors% < threshold (in PERCENT)
};
inline constexpr ProgressivePolicy Progressive_Default{
    /*enabled                    */ false,
    /*sliceMin                   */ 8u,
    /*sliceMaxPct                */ 0.50,   // 50% of maxIterCap
    /*deviceDebugLog             */ false,
    /*stopThresholdSurvivorsPct  */ 0.50    // 0.5%
};
} // namespace NacktmullSettings

namespace prog {

// -------------------------- Device-side tiny formatter ------------------------
// ASCII-only helpers for device logs (no snprintf in device code).
static __device__ __forceinline__ int dev_append_lit(char* dst, int pos, int cap, const char* lit) {
    while (*lit && pos < cap - 1) dst[pos++] = *lit++;
    return pos;
}
static __device__ __forceinline__ int dev_append_uint(char* dst, int pos, int cap, unsigned int v) {
    char tmp[16]; int n = 0;
    do { tmp[n++] = char('0' + (v % 10)); v /= 10; } while (v && n < (int)sizeof(tmp));
    for (int i = n - 1; i >= 0 && pos < cap - 1; --i) dst[pos++] = tmp[i];
    return pos;
}
static __device__ __forceinline__ int dev_append_int(char* dst, int pos, int cap, int v) {
    if (v < 0) { if (pos < cap - 1) dst[pos++] = '-'; unsigned int uv = (unsigned int)(-v); return dev_append_uint(dst, pos, cap, uv); }
    return dev_append_uint(dst, pos, cap, (unsigned int)v);
}

// ------------------------------- Device kernels -------------------------------

__global__ __launch_bounds__(256)
void k_reset_state(float2* __restrict__ z,
                   uint32_t* __restrict__ it,
                   uint8_t* __restrict__ flags,
                   uint32_t* __restrict__ esc,
                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = make_float2(0.f, 0.f);
    it[i] = 0u;
    flags[i] = 0u;
    esc[i] = 0u;
}

__global__ __launch_bounds__(256)
void k_progressive_step(float2* __restrict__ z,
                        uint32_t* __restrict__ it,
                        uint8_t* __restrict__ flags,
                        uint32_t* __restrict__ esc,
                        uint32_t* __restrict__ activeCount,
                        int width, int height,
                        float x0, float y0, float dx, float dy,
                        uint32_t addIter, uint32_t maxIterCap, float bailout2,
                        int debugDevice)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = y * width + x;

    float2 zi = z[i];
    uint32_t iters = it[i];
    uint8_t  fl = flags[i];

    // finished (escaped or capped)
    if (fl & 0x03u) return;

    const float cr = x0 + dx * (float)x;
    const float ci = y0 + dy * (float)y;

    // chunked inner loop
    uint32_t local = 0u;
    while (local < addIter && iters < maxIterCap) {
        // z = z^2 + c (use 1 FMA for 2xy)
        const float x2 = zi.x * zi.x;
        const float y2 = zi.y * zi.y;
        const float xy = zi.x * zi.y;
        zi = make_float2((x2 - y2) + cr, fmaf(2.f, xy, ci));

        ++iters;
        ++local;

        // escape?
        const float r2 = zi.x * zi.x + zi.y * zi.y;
        if (r2 > bailout2) {
            fl |= 0x01u;          // escaped
            esc[i] = iters;
            break;
        }
    }

    if (iters >= maxIterCap) {
        fl |= 0x02u;              // reached cap
    }

    z[i] = zi;
    it[i] = iters;
    flags[i] = fl;

    // survivor counting + optional tiny device log
    if ((fl & 0x03u) == 0u) {
        atomicAdd(activeCount, 1u);
        if (debugDevice) {
            char msg[128];
            int p = 0;
            p = dev_append_lit (msg, p, (int)sizeof(msg), "[DEV] survivor x=");
            p = dev_append_int (msg, p, (int)sizeof(msg), x);
            p = dev_append_lit (msg, p, (int)sizeof(msg), " y=");
            p = dev_append_int (msg, p, (int)sizeof(msg), y);
            p = dev_append_lit (msg, p, (int)sizeof(msg), " it=");
            p = dev_append_uint(msg, p, (int)sizeof(msg), iters);
            p = dev_append_lit (msg, p, (int)sizeof(msg), " add=");
            p = dev_append_uint(msg, p, (int)sizeof(msg), local);
            msg[(p < (int)sizeof(msg) ? p : (int)sizeof(msg) - 1)] = '\0';
            LUCHS_LOG_DEVICE(msg);
        }
    }
}

// ------------------------------- Host helpers --------------------------------

static inline dim3 chooseBlock() { return dim3(32, 8, 1); }
static inline dim3 chooseGrid(int w, int h, dim3 b) {
    return dim3((w + (int)b.x - 1) / (int)b.x, (h + (int)b.y - 1) / (int)b.y, 1);
}

struct PixelMap { float x0, y0, dx, dy; };
static inline PixelMap makePixelMap(const ViewportParams& vp)
{
    const double dy = vp.scale / (double)vp.height;
    const double dx = dy;
    const double x0 = vp.centerX - dx * (double)vp.width  * 0.5;
    const double y0 = vp.centerY - dy * (double)vp.height * 0.5;
    return PixelMap{ (float)x0, (float)y0, (float)dx, (float)dy };
}

// ------------------------------ RAII methods ---------------------------------

CudaProgressiveState::~CudaProgressiveState()
{
    if (d_z_)            cudaFree(d_z_);
    if (d_it_)           cudaFree(d_it_);
    if (d_flags_)        cudaFree(d_flags_);
    if (d_escapeIter_)   cudaFree(d_escapeIter_);
    if (d_activeCount_)  cudaFree(d_activeCount_);
}

void CudaProgressiveState::ensure(int width, int height)
{
    if (width == width_ && height == height_) return;

    if (d_z_)            CUDA_CHECK(cudaFree(d_z_));
    if (d_it_)           CUDA_CHECK(cudaFree(d_it_));
    if (d_flags_)        CUDA_CHECK(cudaFree(d_flags_));
    if (d_escapeIter_)   CUDA_CHECK(cudaFree(d_escapeIter_));
    if (d_activeCount_)  CUDA_CHECK(cudaFree(d_activeCount_));

    width_ = width;
    height_ = height;
    const size_t n = (size_t)width_ * (size_t)height_;

    CUDA_CHECK(cudaMalloc(&d_z_,           n * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_it_,          n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_flags_,       n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_escapeIter_,  n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_activeCount_, sizeof(uint32_t)));

    reset(0);

    LUCHS_LOG_HOST("[PROG] ensure buffers width=%d height=%d bytes=%.1fMB",
                   width_, height_,
                   (n*(sizeof(float2)+sizeof(uint32_t)+sizeof(uint8_t)+sizeof(uint32_t)) + sizeof(uint32_t)) / (1024.0*1024.0));
}

void CudaProgressiveState::reset(cudaStream_t stream)
{
    const int n = width_ * height_;
    if (n <= 0) return;
    dim3 block(256);
    dim3 grid((n + (int)block.x - 1) / (int)block.x);
    k_reset_state<<<grid, block, 0, stream>>>(d_z_, d_it_, d_flags_, d_escapeIter_, n);
    CUDA_CHECK(cudaGetLastError());
    const uint32_t zero = 0u;
    CUDA_CHECK(cudaMemcpyAsync(d_activeCount_, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
}

void CudaProgressiveState::maybeResetOnChange(const ViewportParams& vp, bool enableReset, cudaStream_t stream)
{
    if (!enableReset) { lastCx_ = vp.centerX; lastCy_ = vp.centerY; lastScale_ = vp.scale; return; }

    if (vp.width != width_ || vp.height != height_ ||
        vp.centerX != lastCx_ || vp.centerY != lastCy_ || vp.scale != lastScale_) {
        reset(stream);
        LUCHS_LOG_HOST("[PROG] reset-on-change cx=%.17g cy=%.17g scale=%.17g w=%d h=%d",
                       vp.centerX, vp.centerY, vp.scale, vp.width, vp.height);
    }
    lastCx_ = vp.centerX; lastCy_ = vp.centerY; lastScale_ = vp.scale;
}

ProgressiveMetrics CudaProgressiveState::step(const ViewportParams& vp, const ProgressiveConfig& cfg, cudaStream_t stream)
{
    ensure(vp.width, vp.height);
    maybeResetOnChange(vp, cfg.resetOnChange, stream);

    const uint32_t zero = 0u;
    CUDA_CHECK(cudaMemcpyAsync(d_activeCount_, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    const auto map = makePixelMap(vp);
    const dim3 block = chooseBlock();
    const dim3 grid  = chooseGrid(width_, height_, block);

    // --- Optional progressive policy (default OFF → neutral) ------------------
    constexpr auto P = NacktmullSettings::Progressive_Default;
    const uint32_t sliceMin = P.enabled ? std::max<uint32_t>(1u, P.sliceMin) : 1u;
    const uint32_t sliceMax = P.enabled ? std::max<uint32_t>(1u, (uint32_t)std::floor(P.sliceMaxPct * (double)cfg.maxIterCap))
                                        : cfg.maxIterCap;
    const uint32_t effAddIter = std::min<uint32_t>(std::max<uint32_t>(cfg.chunkIter, sliceMin), sliceMax);
    const int devDbg = (cfg.debugDevice || (P.enabled && P.deviceDebugLog)) ? 1 : 0;
    // --------------------------------------------------------------------------

    float ms = 0.0f;

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        cudaEvent_t evStart{}, evStop{};
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));
        CUDA_CHECK(cudaEventRecord(evStart, stream));

        k_progressive_step<<<grid, block, 0, stream>>>(
            d_z_, d_it_, d_flags_, d_escapeIter_, d_activeCount_,
            width_, height_,
            map.x0, map.y0, map.dx, map.dy,
            effAddIter, cfg.maxIterCap, cfg.bailout2,
            devDbg
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(evStop, stream));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
    } else {
        k_progressive_step<<<grid, block, 0, stream>>>(
            d_z_, d_it_, d_flags_, d_escapeIter_, d_activeCount_,
            width_, height_,
            map.x0, map.y0, map.dx, map.dy,
            effAddIter, cfg.maxIterCap, cfg.bailout2,
            devDbg
        );
        CUDA_CHECK(cudaGetLastError());
    }

    uint32_t still = 0u;
    CUDA_CHECK(cudaMemcpy(&still, d_activeCount_, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    ProgressiveMetrics m;
    m.kernel_ms = ms;
    m.stillActive = still;
    m.addIterApplied = effAddIter;

    const double totalPx = (double)width_ * (double)height_;
    const double survivorsPct = (totalPx > 0.0) ? (100.0 * (double)still / totalPx) : 0.0;

    if constexpr (Settings::debugLogging) {
        if (P.enabled) {
            LUCHS_LOG_HOST("[PROG] cfg clamp addIter=%u->%u sliceMin=%u sliceMax=%u surv=%.3f%%",
                           cfg.chunkIter, effAddIter, sliceMin, sliceMax, (float)survivorsPct);
        }
    }

    // Optional: Schwellenhinweis (kein API-Wechsel, nur Log-Hinweis)
    if (P.enabled && survivorsPct < P.stopThresholdSurvivorsPct) {
        LUCHS_LOG_HOST("[PROG] HALT-SUGGEST thresh=%.3f%% survivors=%.3f%% w=%d h=%d",
                       (float)P.stopThresholdSurvivorsPct, (float)survivorsPct, width_, height_);
    }

    LUCHS_LOG_HOST("[PROG] step done addIter=%u maxCap=%u ms=%.3f survivors=%u w=%d h=%d",
                   m.addIterApplied, cfg.maxIterCap, m.kernel_ms, m.stillActive, width_, height_);

    return m;
}

} // namespace prog
