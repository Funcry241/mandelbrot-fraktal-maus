///// Otter: Hot path: fence poll, PBO map/unmap, kernel launch, E/C transfers, telemetry.
///// Schneefuchs: Shared state via Detail; ring-size asserts; numeric CUDA rc logs.
///// Maus: On any CUDA error: immediate device-log flush; one ASCII line per event.
///// Datei: src/cuda_interop_render.cu

#include "pch.hpp"

#include "cuda_interop.hpp"
#include "cuda_interop_state.hpp"

#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"   // LuchsLogger::flushDeviceLogToHost
#include "core_kernel.h"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#if !defined(__CUDA_ARCH__)
  #include <chrono>
  #include <iomanip>
#endif

// ---- Minimal GL forward decls -----------------------------------------------
struct __GLsync; using GLsync = __GLsync*;
#if !defined(__gl_h_) && !defined(GLEW_H) && !defined(__glew_h__)
  using GLuint     = unsigned int;
  using GLenum     = unsigned int;
  using GLbitfield = unsigned int;
  using GLuint64   = unsigned long long;
  extern "C" {
      GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);
      void   glDeleteSync(GLsync sync);
  }
#endif

// ---- Kernel (extern "C") ----------------------------------------------------
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, uint16_t* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int tile,
    cudaStream_t stream
);

// PERT telemetry (updated by render kernel)
extern __device__ float d_deltaMax;

// PERT param setter (defined in src/nacktmull.cu)
extern "C" void nacktmull_set_perturb(const PerturbParams& p, const double2* zrefGlobalDev);

namespace CudaInterop {
using namespace Detail;

// Single source of truth for ring size
static constexpr int kRing = Settings::pboRingSize;
static_assert(kRing > 0, "Settings::pboRingSize must be > 0");
static_assert(RendererState::kPboRingSize == Settings::pboRingSize,
              "RendererState::kPboRingSize must match Settings::pboRingSize");

// Pointer-Typ exakt an s_pboActive koppeln → kein Namespace-Mismatch mehr
using PBOResPtr = decltype(s_pboActive);

struct MapGuard {
    PBOResPtr r = nullptr;
    void*   ptr   = nullptr;
    size_t  bytes = 0;
    explicit MapGuard(PBOResPtr rr) : r(rr) { if (r) { ptr = r->mapAndLog(bytes); } }
    ~MapGuard() { if (r) r->unmap(); }
    MapGuard(const MapGuard&) = delete;
    MapGuard& operator=(const MapGuard&) = delete;
};

void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
    int width, int height,
    float zoom, float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float2& newOffset, bool& shouldZoom,
    int tileSize, RendererState& state,
    cudaStream_t renderStream,
    cudaStream_t copyStream
){
#if !defined(__CUDA_ARCH__)
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs = 0.0, mbMs = 0.0;
#endif

    if (!s_pboActive) throw std::runtime_error("[FATAL] CUDA PBO not registered!");
    if (width <= 0 || height <= 0)  throw std::runtime_error("invalid framebuffer dims");
    if (tileSize <= 0) {
        int was = tileSize;
        tileSize = Settings::BASE_TILE_SIZE > 0 ? Settings::BASE_TILE_SIZE : 16;
        LUCHS_LOG_HOST("[WARN] tileSize<=0 (%d) -> using %d", was, tileSize);
    }

    const size_t totalPx  = size_t(width) * size_t(height);
    const int    tilesX   = (width  + tileSize - 1) / tileSize;
    const int    tilesY   = (height + tileSize - 1) / tileSize;
    const int    numTiles = tilesX * tilesY;

    const size_t itBytes = totalPx * sizeof(uint16_t);
    const size_t enBytes = size_t(numTiles) * sizeof(float);
    const size_t ctBytes = size_t(numTiles) * sizeof(float);

    if (d_iterations.size() < itBytes || d_entropy.size() < enBytes || d_contrast.size() < ctBytes)
        throw std::runtime_error("CudaInterop::renderCudaFrame: device buffers undersized");

    try {
        // Pre-map fence poll (non-blocking). If busy, skip this frame's upload deterministically.
        {
            GLsync f = state.pboFence[state.pboIndex];
            if (f) {
                GLenum r = glClientWaitSync(f, 0, 0);
                if (r == 0x9111 /*GL_TIMEOUT_EXPIRED*/ || r == 0 /*GL_WAIT_FAILED (conservative)*/) {
                    state.skipUploadThisFrame = true;
                    if (state.pboIndex >= 0 && state.pboIndex < kRing) {
                        state.ringSkip++;
                    }
                    if constexpr (Settings::debugLogging) {
                        LUCHS_LOG_HOST("[ZK][UP] pre-map fence busy -> skip upload this frame (ring=%d/%d)",
                                       state.pboIndex, kRing);
                    }
                    return;
                }
                glDeleteSync(f);
                state.pboFence[state.pboIndex] = 0;
            }
        }

    #if !defined(__CUDA_ARCH__)
        const auto tMap0 = std::chrono::high_resolution_clock::now();
    #endif
        MapGuard map(s_pboActive);
        if (!map.ptr) throw std::runtime_error("pboResource->map() returned null");

    #if !defined(__CUDA_ARCH__)
        const auto tMap1 = std::chrono::high_resolution_clock::now();
        mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
    #endif

        const size_t needBytes = size_t(width) * size_t(height) * sizeof(uchar4);
        if (map.bytes < needBytes) throw std::runtime_error("PBO byte size mismatch");

        if (state.pboIndex >= 0 && state.pboIndex < kRing) {
            state.ringUse[state.pboIndex]++;
        }

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PBO][PTR] ptr=%p bytes=%zu need=%zu", map.ptr, map.bytes, needBytes);
        }

        ensureEventsOnce();
        (void)cudaGetLastError();

        // PERT: clear d_deltaMax if perturbation active
        const bool pertActive = (state.zrefCount > 0);
        static void* s_deltaMaxDev = nullptr;
        if (pertActive) {
            if (!s_deltaMaxDev) { CUDA_CHECK(cudaGetSymbolAddress(&s_deltaMaxDev, d_deltaMax)); }
            CUDA_CHECK(cudaMemsetAsync(s_deltaMaxDev, 0, sizeof(float), renderStream));
        } else {
            state.deltaMaxLast = 0.0;
        }

        // PERT: set parameters (global zref if selected)
        {
            PerturbParams p{};
            p.active     = (state.zrefCount > 0) ? 1 : 0;
            p.len        = state.zrefCount;
            p.segSize    = state.zrefSegSize;
            p.store      = state.perturbStore;
            p.c_ref      = state.c_ref;
            p.deltaGuard = Settings::deltaGuardAbs;
            p.version    = state.zrefVersion;

            const double2* zrefPtr = nullptr;
            if (p.active && p.store == PertStore::Global) {
                zrefPtr = static_cast<const double2*>(state.d_zrefGlobal.get());
            }
            nacktmull_set_perturb(p, zrefPtr);
        }

        CUDA_CHECK(cudaEventRecord(s_evStart, renderStream));

        launch_mandelbrotHybrid(static_cast<uchar4*>(map.ptr),
                                static_cast<uint16_t*>(d_iterations.get()),
                                width, height, zoom, offset, maxIterations, tileSize,
                                renderStream);
        cudaError_t mbErrLaunch = cudaGetLastError();

        CUDA_CHECK(cudaEventRecord(s_evStop, renderStream));
        cudaError_t mbErrSync = cudaEventSynchronize(s_evStop);

    #if !defined(__CUDA_ARCH__)
        if (mbErrSync == cudaSuccess) {
            float ms = 0.0f; cudaEventElapsedTime(&ms, s_evStart, s_evStop); mbMs = ms;
        }
    #endif
        if (mbErrLaunch != cudaSuccess || mbErrSync != cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA][ERR] mandelbrot rc_launch=%d rc_sync=%d", (int)mbErrLaunch, (int)mbErrSync);
            LuchsLogger::flushDeviceLogToHost(0);
            throw std::runtime_error("CUDA failure: mandelbrot kernel");
        }

        // PERT: mirror deltaMax back to host
        if (pertActive) {
            float h_deltaMax = 0.0f;
            CUDA_CHECK(cudaStreamWaitEvent(copyStream, s_evStop, 0));
            CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_deltaMax, d_deltaMax, sizeof(float), 0, cudaMemcpyDeviceToHost, copyStream));
            CUDA_CHECK(cudaStreamSynchronize(copyStream));
            state.deltaMaxLast = (double)h_deltaMax;
        }

        // Entropy/Contrast on render stream; host copies on copy stream (after event)
        ::computeCudaEntropyContrast(
            static_cast<const uint16_t*>(d_iterations.get()),
            static_cast<float*>(d_entropy.get()),
            static_cast<float*>(d_contrast.get()),
            width, height, tileSize, maxIterations,
            renderStream,
            state.evEcDone
        );

        if (h_entropy.capacity()  < size_t(numTiles)) h_entropy.reserve(size_t(numTiles));
        if (h_contrast.capacity() < size_t(numTiles)) h_contrast.reserve(size_t(numTiles));
        h_entropy.resize(size_t(numTiles));
        h_contrast.resize(size_t(numTiles));

        if (state.evEcDone) {
            CUDA_CHECK(cudaStreamWaitEvent(copyStream, state.evEcDone, 0));
        }

        // WICHTIG: cudaMemcpyAsync mit Kind + Stream
        CUDA_CHECK(cudaMemcpyAsync(h_entropy.data(),  d_entropy.get(),  enBytes, cudaMemcpyDeviceToHost, copyStream));
        CUDA_CHECK(cudaMemcpyAsync(h_contrast.data(), d_contrast.get(), ctBytes,  cudaMemcpyDeviceToHost, copyStream));

        if (state.evCopyDone) {
            CUDA_CHECK(cudaEventRecord(state.evCopyDone, copyStream));
        }

        shouldZoom = false; newOffset = offset;

    #if !defined(__CUDA_ARCH__)
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        state.lastTimings.valid            = true;
        state.lastTimings.pboMap           = mapMs;
        state.lastTimings.mandelbrotTotal  = mbMs;
        state.lastTimings.mandelbrotLaunch = 0.0;
        state.lastTimings.mandelbrotSync   = 0.0;
        state.lastTimings.entropy          = 0.0;
        state.lastTimings.contrast         = 0.0;
        state.lastTimings.deviceLogFlush   = 0.0;

        if constexpr (Settings::performanceLogging)
            LUCHS_LOG_HOST("[PERF][ZK] map=%.2f mandelbrot=%.2f entropy=%.2f contrast=%.2f total=%.2f",
                           mapMs, mbMs, state.lastTimings.entropy, state.lastTimings.contrast, totalMs);
    #endif
    } catch (...) {
        LuchsLogger::flushDeviceLogToHost(0);
        throw;
    }
}

} // namespace CudaInterop
