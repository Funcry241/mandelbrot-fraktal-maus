// Datei: src/cuda_interop.cu
// 🐜 Ameise: deterministisches Logging & robuste Pfade.
// 🦦 Otter: Nacktmull-only (Host-Iters + GPU-Shade), aber mit Boot-Fastpath.
// 🦊 Schneefuchs: Früher Fallback, Iter-Budgetierung, klare STEP-Logs.

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

namespace CudaInterop {

static bear_CudaPBOResource* pboResource      = nullptr;
static bool  pauseZoom                        = false;
static bool  s_deviceInitDone                 = false;

static void*  s_hostRegEntropyPtr   = nullptr;
static size_t s_hostRegEntropyBytes = 0;
static void*  s_hostRegContrastPtr  = nullptr;
static size_t s_hostRegContrastBytes= 0;

// ---- Boot-Fallback: sofort zeichnen, noch **vor** Host-Iterationen ----------
static int  s_bootFrames             = 0;
static constexpr int kBootFallbackFrames = 2;

// Winziger Kernel für harten Fill (Magenta) im Fehlerfall
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
    if (cap == 0) { if (regPtr) { CUDA_CHECK(cudaHostUnregister(regPtr)); regPtr=nullptr; regBytes=0; } return; }
    void* ptr = static_cast<void*>(vec.data());
    const size_t bytes = cap * sizeof(float);
    if (ptr != regPtr || bytes != regBytes) {
        if (regPtr) CUDA_CHECK(cudaHostUnregister(regPtr));
        CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
        regPtr  = ptr; regBytes = bytes;
        if constexpr (Settings::debugLogging) { LUCHS_LOG_HOST("[PIN] host-register ptr=%p bytes=%zu", ptr, bytes); }
    }
}

void registerPBO(const Hermelin::GLBuffer& pbo) {
    if (pboResource) { if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[ERROR] registerPBO: already registered!"); return; }
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
    double mapMs=0.0, entMs=0.0, conMs=0.0, hostItMs=0.0, shadeMs=0.0;
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

    // ---------- BOOT-FASTPATH (ohne Host-Iterations) ----------
    if (s_bootFrames < kBootFallbackFrames) {
        if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[BOOT FASTPATH] draw test pattern before host iterations (frame=%d)", s_bootFrames);

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
            throw std::runtime_error("PBO byte size mismatch (boot fastpath)");
        }

        dim3 block(32,8), grid((width+31)/32, (height+7)/8);
        shade_test_pattern<<<grid,block>>>(devSurface, width, height, 24);
        CUDA_CHECK(cudaGetLastError());
        pboResource->unmap();

        // Host-Arrays auf 0 setzen (keine Analyse im Boot)
        h_entropy.assign(size_t(numTiles), 0.0f);
        h_contrast.assign(size_t(numTiles), 0.0f);
        shouldZoom = false; newOffset = offset;

        ++s_bootFrames;

        state.lastTimings.valid            = true;
        state.lastTimings.pboMap           = mapMs;
        state.lastTimings.mandelbrotTotal  = 0.0;
        state.lastTimings.mandelbrotLaunch = 0.0;
        state.lastTimings.mandelbrotSync   = 0.0;
        state.lastTimings.entropy          = 0.0;
        state.lastTimings.contrast         = 0.0;
        state.lastTimings.deviceLogFlush   = 0.0;
        return;
    }

    // ---------- Host-Iterations mit Budget ----------
    // Worst-case Work-Schätzung und Budgetierung
    const unsigned long long work = (unsigned long long)totalPixels * (unsigned long long)maxIterations;
    static constexpr unsigned long long WORK_LIMIT = 200ull * 1000ull * 1000ull; // ~2e8 "Iteration-Schritte"
    int iterCap = maxIterations;
    if (work > WORK_LIMIT) {
        iterCap = int(std::max<unsigned long long>(1ull, WORK_LIMIT / std::max<unsigned long long>(1ull, totalPixels)));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[STEP] A0: cap maxIterations %d -> %d (work=%llu, limit=%llu)",
                           maxIterations, iterCap, work, WORK_LIMIT);
        }
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[STEP] A: host iterations start (w=%d h=%d itMax=%d zoom=%.6f off=(%.6f,%.6f))",
                       width, height, iterCap, zoom, (double)offset.x, (double)offset.y);
    }

    std::vector<int> h_iters(totalPixels);

#ifndef CUDA_ARCH
    hostItMs = Nacktmull::compute_host_iterations(
        width, height, (double)zoom, (double)offset.x, (double)offset.y, iterCap, h_iters);
#else
    (void)Nacktmull::compute_host_iterations(
        width, height, (double)zoom, (double)offset.x, (double)offset.y, iterCap, h_iters);
#endif
    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] A-done: host iterations ready");

    bool anyOutside = false;
    for (size_t i=0;i<h_iters.size();++i){ if (h_iters[i] < iterCap) { anyOutside = true; break; } }
    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] A2: anyOutside=%d", anyOutside?1:0);

    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] B: memcpy H2D (iterations)");
    CUDA_CHECK(cudaMemcpy(d_iterations.get(), h_iters.data(), it_bytes, cudaMemcpyHostToDevice));

#ifndef CUDA_ARCH
    const auto tMap0 = std::chrono::high_resolution_clock::now();
#endif
    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] C: map PBO");
    size_t surfBytes=0;
    uchar4* devSurface = static_cast<uchar4*>(pboResource->mapAndLog(surfBytes));
#ifndef CUDA_ARCH
    const auto tMap1 = std::chrono::high_resolution_clock::now();
    mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
#endif
    if (!devSurface) throw std::runtime_error("pboResource->map() returned null");
    const size_t expected = size_t(width) * size_t(height) * sizeof(uchar4);
    if (surfBytes < expected) { pboResource->unmap(); throw std::runtime_error("PBO byte size mismatch"); }

    float shadeMsEv = 0.0f; cudaEvent_t evS0=nullptr, evS1=nullptr;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&evS0)); CUDA_CHECK(cudaEventCreate(&evS1)); CUDA_CHECK(cudaEventRecord(evS0,0));
    }

    dim3 block(32,8), grid((width+31)/32, (height+7)/8);
    if (anyOutside) {
        if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] D1: shade_from_iterations");
        shade_from_iterations<<<grid,block>>>(
            devSurface, static_cast<const int*>(d_iterations.get()), width, height, iterCap);
    } else {
        if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] D2: test pattern (all inside)");
        shade_test_pattern<<<grid,block>>>(devSurface, width, height, 24);
    }

    cudaError_t shadeErr = cudaGetLastError();
    if (shadeErr != cudaSuccess) {
        LUCHS_LOG_HOST("[FATAL] shade kernel failed err=%d -> filling PBO magenta", (int)shadeErr);
        const uchar4 mag = make_uchar4(255,0,255,255);
        fill_rgba_kernel<<<grid,block>>>(devSurface, width, height, mag);
        CUDA_CHECK(cudaGetLastError());
    }

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventRecord(evS1,0)); CUDA_CHECK(cudaEventSynchronize(evS1));
        CUDA_CHECK(cudaEventElapsedTime(&shadeMsEv, evS0, evS1));
        CUDA_CHECK(cudaEventDestroy(evS0));  CUDA_CHECK(cudaEventDestroy(evS1));
#ifndef CUDA_ARCH
        shadeMs = (double)shadeMsEv;
#endif
    }

#ifndef CUDA_ARCH
    const auto tEC0 = std::chrono::high_resolution_clock::now();
#endif
    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] E: computeCudaEntropyContrast");
    ::computeCudaEntropyContrast(
        static_cast<const int*>(d_iterations.get()),
        static_cast<float*>(d_entropy.get()),
        static_cast<float*>(d_contrast.get()),
        width, height, tileSize, iterCap);
#ifndef CUDA_ARCH
    const auto tEC1 = std::chrono::high_resolution_clock::now();
    const double ecMs = std::chrono::duration<double, std::milli>(tEC1 - tEC0).count();
    entMs = ecMs * 0.5; conMs = ecMs * 0.5;
#endif

    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] F: prepare host vectors + pin");
    if (h_entropy.capacity()  < size_t(numTiles)) h_entropy.reserve(size_t(numTiles));
    if (h_contrast.capacity() < size_t(numTiles)) h_contrast.reserve(size_t(numTiles));
    ensureHostPinned(h_entropy,  s_hostRegEntropyPtr,  s_hostRegEntropyBytes);
    ensureHostPinned(h_contrast, s_hostRegContrastPtr, s_hostRegContrastBytes);

    h_entropy.resize(size_t(numTiles));
    h_contrast.resize(size_t(numTiles));

    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] G: memcpy D2H (entropy/contrast)");
    CUDA_CHECK(cudaMemcpy(h_entropy.data(),  d_entropy.get(),  entropy_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast.get(), contrast_bytes, cudaMemcpyDeviceToHost));

    shouldZoom = false; newOffset = offset;

    if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[STEP] H: unmap PBO");
    pboResource->unmap();

#ifndef CUDA_ARCH
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    state.lastTimings.valid            = true;
#ifndef CUDA_ARCH
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = shadeMs;
    state.lastTimings.mandelbrotLaunch = 0.0;
    state.lastTimings.mandelbrotSync   = 0.0;
    state.lastTimings.entropy          = entMs;
    state.lastTimings.contrast         = conMs;
    state.lastTimings.deviceLogFlush   = 0.0;
#else
    state.lastTimings = {};
#endif

#ifndef CUDA_ARCH
    if constexpr (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] path=test mp=%.2f hostIt=%.2f shade=%.2f en=%.2f ct=%.2f tt=%.2f",
                       mapMs, hostItMs, shadeMs, entMs, conMs, totalMs);
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
    if (msg) { if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[CHECK] cudaGetErrorString(dummy) = \"%s\"", msg); return true; }
    LUCHS_LOG_HOST("[FATAL] cudaGetErrorString returned null"); return false;
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
