///// Otter: Phase A – enforce GPU iteration; fix link by matching extern "C" for launch_mandelbrotHybrid.
///// Schneefuchs: Map/Unmap timings, compact [PERF] summary; WriteDiscard via bear_CudaPBOResource policy.
///// Maus: Deterministic orchestrator; CPU-iteration path removed; ASCII-only logs.

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"
#include "nacktmull_shade.cuh"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
#include <stdexcept>
#include <cstdint>

#ifndef CUDA_ARCH
  #include <chrono>
#endif

#include "nacktmull_anchor.hpp"
#include "nacktmull_host.hpp"

// 🦦 Otter: core_kernel.cu defines the host launch with C linkage; match here to avoid LNK2019.
extern "C" void launch_mandelbrotHybrid(
    uchar4* surface,
    int width, int height,
    float zoom, float2 offset,
    int maxIterations,
    int* d_iterations,
    int tileSize
);

namespace CudaInterop {

static bear_CudaPBOResource* pboResource      = nullptr;
static bool  pauseZoom                        = false;
static bool  s_deviceInitDone                 = false;

static void*  s_hostRegEntropyPtr   = nullptr;
static size_t s_hostRegEntropyBytes = 0;
static void*  s_hostRegContrastPtr  = nullptr;
static size_t s_hostRegContrastBytes= 0;

// Minimal kernel for emergency fill (magenta) if shading fails
static __global__ void fill_rgba_kernel(uchar4* dst, int w, int h, uchar4 c) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[y * w + x] = c;
}

static inline void ensureDeviceOnce() {
    if (!s_deviceInitDone) { CUDA_CHECK(cudaSetDevice(0)); s_deviceInitDone = true; }
}

static inline void ensureHostPinned(std::vector<float>& vec, void*& regPtr, size_t& regBytes) {
    const size_t cap = vec.capacity();
    if (cap == 0) {
        if (regPtr) { CUDA_CHECK(cudaHostUnregister(regPtr)); regPtr=nullptr; regBytes=0; }
        return;
    }
    void* ptr = static_cast<void*>(vec.data());
    const size_t bytes = cap * sizeof(float);
    if (ptr != regPtr || bytes != regBytes) {
        if (regPtr) CUDA_CHECK(cudaHostUnregister(regPtr));
        CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
        regPtr  = ptr; regBytes = bytes;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] host-register ptr=%p bytes=%zu", ptr, bytes);
        }
    }
}

void registerPBO(const Hermelin::GLBuffer& pbo) {
    if (pboResource) {
        if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[ERROR] registerPBO: already registered!");
        return;
    }
    ensureDeviceOnce();

    GLint prev=0; glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.id());
    GLint now=0; glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &now);
    if (now != (GLint)pbo.id()) {
        LUCHS_LOG_HOST("[FATAL] GL bind failed - buffer %u was not bound (GL says %d)", pbo.id(), now);
        throw std::runtime_error("glBindBuffer(GL_PIXEL_UNPACK_BUFFER) failed");
    }

    pboResource = new bear_CudaPBOResource(pbo.id());

    size_t warm=0;
    if (auto* ptr = pboResource->mapAndLog(warm)) {
        (void)ptr; pboResource->unmap();
        if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[PBO] warm-up map/unmap done (%zu bytes)", warm);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, (GLuint)prev);
}

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
    int tileSize, RendererState& state
) {
#ifndef CUDA_ARCH
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs=0.0, mbMs=0.0, entMs=0.0, conMs=0.0;
#endif

    if (!pboResource) throw std::runtime_error("[FATAL] CUDA PBO not registered!");

    const size_t totalPixels = size_t(width) * size_t(height);
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    const size_t it_bytes       = totalPixels * sizeof(int);
    const size_t entropy_bytes  = size_t(numTiles) * sizeof(float);
    const size_t contrast_bytes = size_t(numTiles) * sizeof(float);

    if (d_iterations.size() < it_bytes || d_entropy.size() < entropy_bytes || d_contrast.size() < contrast_bytes) {
        LUCHS_LOG_HOST("[FATAL] device buffers too small: it=%zu/%zu en=%zu/%zu ct=%zu/%zu",
                       d_iterations.size(), it_bytes, d_entropy.size(), entropy_bytes, d_contrast.size(), contrast_bytes);
        throw std::runtime_error("CudaInterop::renderCudaFrame: device buffers undersized");
    }

#ifndef CUDA_ARCH
    const auto tMap0 = std::chrono::high_resolution_clock::now();
#endif
    size_t surfBytes=0;
    uchar4* devSurface = static_cast<uchar4*>(pboResource->mapAndLog(surfBytes));
#ifndef CUDA_ARCH
    const auto tMap1 = std::chrono::high_resolution_clock::now();
    mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
#endif
    if (!devSurface) throw std::runtime_error("pboResource->map() returned null");
    const size_t expected = size_t(width) * size_t(height) * sizeof(uchar4);
    if (surfBytes < expected) {
        pboResource->unmap();
        throw std::runtime_error("PBO byte size mismatch");
    }

    // === Mandelbrot on GPU (single launch, no CPU iterations) ===
    cudaEvent_t ev0=nullptr, ev1=nullptr;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, 0));
    }

    launch_mandelbrotHybrid(
        devSurface,
        width, height,
        zoom, offset,
        maxIterations,
        static_cast<int*>(d_iterations.get()),
        tileSize
    );

    cudaError_t mbErr = cudaGetLastError();
    if (mbErr != cudaSuccess) {
        LUCHS_LOG_HOST("[FATAL] mandelbrot launch failed err=%d -> filling PBO magenta", (int)mbErr);
        dim3 block(32,8), grid((width+31)/32, (height+7)/8);
        const uchar4 mag = make_uchar4(255,0,255,255);
        fill_rgba_kernel<<<grid,block>>>(devSurface, width, height, mag);
        CUDA_CHECK(cudaGetLastError());
    }

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        mbMs = (double)ms;
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
    }

#ifndef CUDA_ARCH
    const auto tEC0 = std::chrono::high_resolution_clock::now();
#endif
    ::computeCudaEntropyContrast(
        static_cast<const int*>(d_iterations.get()),
        static_cast<float*>(d_entropy.get()),
        static_cast<float*>(d_contrast.get()),
        width, height, tileSize, maxIterations
    );
#ifndef CUDA_ARCH
    const auto tEC1 = std::chrono::high_resolution_clock::now();
    const double ecMs = std::chrono::duration<double, std::milli>(tEC1 - tEC0).count();
    entMs = ecMs * 0.5; conMs = ecMs * 0.5;
#endif

    // Prepare pinned host vectors and copy analysis home
    if (h_entropy.capacity()  < size_t(numTiles)) h_entropy.reserve(size_t(numTiles));
    if (h_contrast.capacity() < size_t(numTiles)) h_contrast.reserve(size_t(numTiles));
    ensureHostPinned(h_entropy,  s_hostRegEntropyPtr,  s_hostRegEntropyBytes);
    ensureHostPinned(h_contrast, s_hostRegContrastPtr, s_hostRegContrastBytes);
    h_entropy.resize(size_t(numTiles));
    h_contrast.resize(size_t(numTiles));

    CUDA_CHECK(cudaMemcpy(h_entropy.data(),  d_entropy.get(),  entropy_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast.get(), contrast_bytes, cudaMemcpyDeviceToHost));

    // No retarget here; let zoom logic decide
    shouldZoom = false;
    newOffset  = offset;

    pboResource->unmap();

#ifndef CUDA_ARCH
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    state.lastTimings.valid            = true;
#ifndef CUDA_ARCH
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = mbMs;
    state.lastTimings.mandelbrotLaunch = 0.0;   // kept for compatibility
    state.lastTimings.mandelbrotSync   = 0.0;   // single implicit sync via memcpy D2H
    state.lastTimings.entropy          = entMs;
    state.lastTimings.contrast         = conMs;
    state.lastTimings.deviceLogFlush   = 0.0;
#else
    state.lastTimings = {};
#endif

#ifndef CUDA_ARCH
    if constexpr (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] path=gpu mp=%.2f mb=%.2f en=%.2f ct=%.2f tt=%.2f",
                       mapMs, mbMs, entMs, conMs, totalMs);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[TIME] total=%.2f", totalMs);
    }
#endif
}

void setPauseZoom(bool pause) { pauseZoom = pause; }
bool getPauseZoom()           { return pauseZoom; }

bool precheckCudaRuntime() {
    int deviceCount = 0;
    cudaError_t e1 = cudaFree(0);
    cudaError_t e2 = cudaGetDeviceCount(&deviceCount);
    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[CUDA] precheck err1=%d err2=%d count=%d", (int)e1, (int)e2, deviceCount);
    return e1 == cudaSuccess && e2 == cudaSuccess && deviceCount > 0;
}

bool verifyCudaGetErrorStringSafe() {
    cudaError_t dummy = cudaErrorInvalidValue;
    const char* msg = cudaGetErrorString(dummy);
    if (msg) {
        if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[CHECK] cudaGetErrorString(dummy) = \"%s\"", msg);
        return true;
    }
    LUCHS_LOG_HOST("[FATAL] cudaGetErrorString returned null");
    return false;
}

void unregisterPBO() {
    if (s_hostRegEntropyPtr)  { cudaHostUnregister(s_hostRegEntropyPtr);  s_hostRegEntropyPtr=nullptr;  s_hostRegEntropyBytes=0; }
    if (s_hostRegContrastPtr) { cudaHostUnregister(s_hostRegContrastPtr); s_hostRegContrastPtr=nullptr; s_hostRegContrastBytes=0; }
    delete pboResource; pboResource = nullptr;
}

void logCudaDeviceContext(const char* tag) {
    int dev=-1; cudaError_t e0=cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaError_t e1=(e0==cudaSuccess && dev>=0) ? cudaGetDeviceProperties(&prop, dev) : cudaErrorInvalidDevice;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        if (e0==cudaSuccess && e1==cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA] ctx tag=%s device=%d name=\"%s\" cc=%d.%d sms=%d vram=%lluMB",
                (tag?tag:"(null)"), dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                (unsigned long long)(prop.totalGlobalMem / (1024ull*1024ull)));
        } else {
            LUCHS_LOG_HOST("[CUDA] ctx tag=%s deviceQuery failed e0=%d e1=%d dev=%d",
                (tag?tag:"(null)"), (int)e0, (int)e1, dev);
        }
    }
}

} // namespace CudaInterop
