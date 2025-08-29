// Datei: src/cuda_interop.cu
// 🐜 Schwarze Ameise: Klare Parametrisierung, deterministisches Logging, robustes Ressourcenhandling.
// 🦦 Otter → Nacktmull-only: Host-Iters + GPU-Shade, kein Legacy-GPU-Path mehr.
// 🦊 Schneefuchs: Transparente Speicher-/Fehlerprüfung. Null Seiteneffekte in Hot-Paths.

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"           // computeCudaEntropyContrast(...)
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"
#include "nacktmull_shade.cuh"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>   // events/memcpy/host register
#include <vector_types.h>   // uchar4
#include <vector>
#include <stdexcept>

#ifndef CUDA_ARCH
  #include <chrono>
#endif

// --------------------------- Nacktmull (immer aktiv) -------------------------
#include "nacktmull_anchor.hpp"    // Nacktmull::compute_host_iterations(...)
#include "nacktmull_host.hpp"      // Host-Iteration

namespace CudaInterop {

// TU-lokaler Zustand
static bear_CudaPBOResource* pboResource      = nullptr;
static bool pauseZoom                         = false;
static bool s_deviceInitDone                  = false;

// Pinned-Host-Registrierung für E/C (schnellere D2H)
static void*  s_hostRegEntropyPtr   = nullptr;
static size_t s_hostRegEntropyBytes = 0;
static void*  s_hostRegContrastPtr  = nullptr;
static size_t s_hostRegContrastBytes= 0;

static inline void ensureDeviceOnce() {
    if (!s_deviceInitDone) {
        CUDA_CHECK(cudaSetDevice(0));
        s_deviceInitDone = true;
    }
}

static inline void ensureHostPinned(std::vector<float>& vec, void*& regPtr, size_t& regBytes) {
    const size_t cap = vec.capacity();
    if (cap == 0) {
        if (regPtr) { CUDA_CHECK(cudaHostUnregister(regPtr)); regPtr = nullptr; regBytes = 0; }
        return;
    }
    void* ptr = static_cast<void*>(vec.data());
    const size_t bytes = cap * sizeof(float);
    if (ptr != regPtr || bytes != regBytes) {
        if (regPtr) CUDA_CHECK(cudaHostUnregister(regPtr));
        CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
        regPtr  = ptr;
        regBytes= bytes;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] host-register ptr=%p bytes=%zu", ptr, bytes);
        }
    }
}

void registerPBO(const Hermelin::GLBuffer& pbo) {
    if (pboResource) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] registerPBO: already registered!");
        }
        return;
    }

    ensureDeviceOnce();

    // Sanity: Bind-State check
    GLint boundBefore = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBefore);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.id());
    GLint boundAfter = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundAfter);

    if (boundAfter != static_cast<GLint>(pbo.id())) {
        LUCHS_LOG_HOST("[FATAL] GL bind failed - buffer %u was not bound (GL reports: %d)", pbo.id(), boundAfter);
        throw std::runtime_error("glBindBuffer(GL_PIXEL_UNPACK_BUFFER) failed");
    }

    pboResource = new bear_CudaPBOResource(pbo.id());

    // 🧊 Warm-up: einmaliges map/unmap, um Treiberpfade zu initialisieren (vermeidet 20ms-Spike im 2. Frame)
    size_t warmBytes = 0;
    if (auto* ptr = pboResource->mapAndLog(warmBytes)) {
        pboResource->unmap();
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PBO] warm-up map/unmap done (%zu bytes)", warmBytes);
        }
    }
}

void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize,
    RendererState& state
) {
#ifndef CUDA_ARCH
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs = 0.0, entMs = 0.0, conMs = 0.0, hostItMs = 0.0, shadeMs = 0.0;
#endif

    if (!pboResource)
        throw std::runtime_error("[FATAL] CUDA PBO not registered!");

    const int totalPixels = width * height;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    // Größencheck
    const size_t it_bytes       = static_cast<size_t>(totalPixels) * sizeof(int);
    const size_t entropy_bytes  = static_cast<size_t>(numTiles)    * sizeof(float);
    const size_t contrast_bytes = static_cast<size_t>(numTiles)    * sizeof(float);

    if (d_iterations.size() < it_bytes ||
        d_entropy.size()    < entropy_bytes ||
        d_contrast.size()   < contrast_bytes)
    {
        LUCHS_LOG_HOST("[FATAL] device buffers too small: it=%zu/%zu en=%zu/%zu ct=%zu/%zu",
                       d_iterations.size(), it_bytes,
                       d_entropy.size(),    entropy_bytes,
                       d_contrast.size(),   contrast_bytes);
        throw std::runtime_error("CudaInterop::renderCudaFrame: device buffers undersized");
    }

    // Deterministische Clears
    CUDA_CHECK(cudaMemset(d_iterations.get(), 0, d_iterations.size()));
    CUDA_CHECK(cudaMemset(d_entropy.get(),    0, d_entropy.size()));
    CUDA_CHECK(cudaMemset(d_contrast.get(),   0, d_contrast.size()));

    // ---------------------------------- NACKTMULL: Host-Iters ----------------------------------
    std::vector<int> h_iters;
    h_iters.resize(static_cast<size_t>(totalPixels));

#ifndef CUDA_ARCH
    hostItMs = Nacktmull::compute_host_iterations(
        width, height,
        static_cast<double>(zoom),
        static_cast<double>(offset.x),
        static_cast<double>(offset.y),
        maxIterations,
        h_iters
    );
#else
    (void)Nacktmull::compute_host_iterations(
        width, height,
        static_cast<double>(zoom),
        static_cast<double>(offset.x),
        static_cast<double>(offset.y),
        maxIterations,
        h_iters
    );
#endif

    // Host → Device (Iterationsbild)
    CUDA_CHECK(cudaMemcpy(d_iterations.get(), h_iters.data(), it_bytes, cudaMemcpyHostToDevice));

    // ---------------------------------- PBO map & GPU-Shade ------------------------------------
#ifndef CUDA_ARCH
    const auto tMap0 = std::chrono::high_resolution_clock::now();
#endif
    size_t surfBytes = 0;
    uchar4* devSurface = static_cast<uchar4*>(pboResource->mapAndLog(surfBytes));
#ifndef CUDA_ARCH
    const auto tMap1 = std::chrono::high_resolution_clock::now();
    mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
#endif
    if (!devSurface) {
        LUCHS_LOG_HOST("[FATAL] surface pointer is null");
        throw std::runtime_error("pboResource->map() returned null");
    }

    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar4);
    if (surfBytes < expected) {
        LUCHS_LOG_HOST("[FATAL] PBO size too small: got=%zu need=%zu (w=%d h=%d)", surfBytes, expected, width, height);
        pboResource->unmap();
        throw std::runtime_error("PBO byte size mismatch");
    }

    // Shade aus Iterationsbild
    float shadeMsEv = 0.0f;
    cudaEvent_t evS0 = nullptr, evS1 = nullptr;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&evS0));
        CUDA_CHECK(cudaEventCreate(&evS1));
        CUDA_CHECK(cudaEventRecord(evS0, 0));
    }

    dim3 block(32, 8);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    shade_from_iterations<<<grid, block>>>(
        devSurface,
        static_cast<const int*>(d_iterations.get()),
        width, height, maxIterations
    );
    CUDA_CHECK(cudaGetLastError());

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventRecord(evS1, 0));
        CUDA_CHECK(cudaEventSynchronize(evS1));
        CUDA_CHECK(cudaEventElapsedTime(&shadeMsEv, evS0, evS1));
        CUDA_CHECK(cudaEventDestroy(evS0));
        CUDA_CHECK(cudaEventDestroy(evS1));
#ifndef CUDA_ARCH
        shadeMs = static_cast<double>(shadeMsEv);
#endif
    }

    // ---------------------------------- Entropie/Kontrast --------------------------------------
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
    entMs = ecMs * 0.5;
    conMs = ecMs * 0.5;
#endif

    // Host-Ziele vorbereiten (keine Reallocs → dann pinnen)
    if ((size_t)h_entropy.capacity()  < (size_t)numTiles) h_entropy.reserve((size_t)numTiles);
    if ((size_t)h_contrast.capacity() < (size_t)numTiles) h_contrast.reserve((size_t)numTiles);
    ensureHostPinned(h_entropy,  s_hostRegEntropyPtr,  s_hostRegEntropyBytes);
    ensureHostPinned(h_contrast, s_hostRegContrastPtr, s_hostRegContrastBytes);

    h_entropy.resize((size_t)numTiles);
    h_contrast.resize((size_t)numTiles);

    CUDA_CHECK(cudaMemcpy(h_entropy.data(),  d_entropy.get(),  entropy_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast.get(), contrast_bytes, cudaMemcpyDeviceToHost));

    // Zoom-Kommunikation (unverändert)
    shouldZoom = false;
    newOffset  = offset;

    pboResource->unmap();

#ifndef CUDA_ARCH
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    // Timings an RendererState (GPU-Anteil = Shade)
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
    state.lastTimings.pboMap           = 0.0;
    state.lastTimings.mandelbrotTotal  = 0.0;
    state.lastTimings.mandelbrotLaunch = 0.0;
    state.lastTimings.mandelbrotSync   = 0.0;
    state.lastTimings.entropy          = 0.0;
    state.lastTimings.contrast         = 0.0;
    state.lastTimings.deviceLogFlush   = 0.0;
#endif

#ifndef CUDA_ARCH
    if constexpr (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] path=nm mp=%.2f hostIt=%.2f shade=%.2f en=%.2f ct=%.2f tt=%.2f",
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
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CUDA] precheck err1=%d err2=%d count=%d", (int)e1, (int)e2, deviceCount);
    }
    return e1 == cudaSuccess && e2 == cudaSuccess && deviceCount > 0;
}

bool verifyCudaGetErrorStringSafe() {
    cudaError_t dummy = cudaErrorInvalidValue;
    const char* msg = cudaGetErrorString(dummy);
    if (msg) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CHECK] cudaGetErrorString(dummy) = \"%s\" -> Aufloesung gefahrlos", msg);
        }
        return true;
    } else {
        LUCHS_LOG_HOST("[FATAL] cudaGetErrorString returned null");
        return false;
    }
}

void unregisterPBO() {
    // Host-Pins sauber lösen
    if (s_hostRegEntropyPtr)  { cudaHostUnregister(s_hostRegEntropyPtr);  s_hostRegEntropyPtr  = nullptr; s_hostRegEntropyBytes  = 0; }
    if (s_hostRegContrastPtr) { cudaHostUnregister(s_hostRegContrastPtr); s_hostRegContrastPtr = nullptr; s_hostRegContrastBytes = 0; }

    delete pboResource;
    pboResource = nullptr;
}

void logCudaDeviceContext(const char* tag)
{
    int dev = -1;
    cudaError_t e0 = cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaError_t e1 = (e0 == cudaSuccess && dev >= 0)
                   ? cudaGetDeviceProperties(&prop, dev)
                   : cudaErrorInvalidDevice;

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        if (e0 == cudaSuccess && e1 == cudaSuccess) {
            // ASCII-only, deterministic
            LUCHS_LOG_HOST("[CUDA] ctx tag=%s device=%d name=\"%s\" cc=%d.%d sms=%d vram=%lluMB",
                (tag ? tag : "(null)"),
                dev,
                prop.name,
                prop.major, prop.minor,
                prop.multiProcessorCount,
                static_cast<unsigned long long>(prop.totalGlobalMem / (1024ull*1024ull))
            );
        } else {
            LUCHS_LOG_HOST("[CUDA] ctx tag=%s deviceQuery failed e0=%d e1=%d dev=%d",
                (tag ? tag : "(null)"), (int)e0, (int)e1, dev);
        }
    }
}

} // namespace CudaInterop
