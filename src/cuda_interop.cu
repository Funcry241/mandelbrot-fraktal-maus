// Datei: src/cuda_interop.cu
// 🐜 Schwarze Ameise: Klare Parametrisierung, deterministisches Logging, robustes Ressourcenhandling.
// 🦦 Otter → Nacktmull-only (JETZT mit schnellem GPU-Iter-Pfad): Iterationen auf der GPU + GPU-Shade.
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
#include "nacktmull_shade.cuh"     // shade_from_iterations(...)
#include "nacktmull_math.cuh"      // pixelToComplex(...)

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>
#include <stdexcept>

#ifndef CUDA_ARCH
  #include <chrono>
#endif

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
}

// -----------------------------------------------------------------------------
// 🐭 Maus FastPath: GPU-Iterationskernel (schnell, blockiert den Host nicht)
//    Füllt d_iterations direkt auf der GPU. Farbgebung erfolgt via shade_from_iterations.
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    float xm = x - 0.25f; float q = xm*xm + y*y;
    if (q*(q + xm) <= 0.25f*y*y) return true;         // main cardioid
    float xp = x + 1.0f; if (xp*xp + y*y <= 0.0625f) return true; // period-2 bulb
    return false;
}

__global__ void maus_iter_kernel(int* __restrict__ itOut,
                                 int w,int h, float zoom, float2 offset, int maxIter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    const int idx = y*w + x;

    const float spanX = 3.5f * (1.0f / zoom);
    const float spanY = spanX * (float)h / (float)w;
    float2 c = pixelToComplex(x + 0.5f, y + 0.5f, w, h, spanX, spanY, offset);

    if (insideMainCardioidOrBulb(c.x, c.y)) { itOut[idx] = maxIter; return; }

    float zx = 0.f, zy = 0.f; int it = 0;
    for (; it < maxIter; ++it){
        float x2 = zx*zx, y2 = zy*zy;
        if (x2 + y2 > 4.f) break;
        float xt = x2 - y2 + c.x; zy = 2.f*zx*zy + c.y; zx = xt;
    }
    itOut[idx] = it;
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
    double mapMs = 0.0, entMs = 0.0, conMs = 0.0, iterMs = 0.0, shadeMs = 0.0;
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

    // --------------------------- GPU-Iterations (Fast Path) ---------------------------
    cudaEvent_t evI0 = nullptr, evI1 = nullptr; float iterMsEv = 0.0f;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&evI0));
        CUDA_CHECK(cudaEventCreate(&evI1));
        CUDA_CHECK(cudaEventRecord(evI0, 0));
    }

    {
        dim3 block(32, 8);
        dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
        maus_iter_kernel<<<grid, block>>>(
            static_cast<int*>(d_iterations.get()),
            width, height, zoom, offset, maxIterations
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventRecord(evI1, 0));
        CUDA_CHECK(cudaEventSynchronize(evI1));
        CUDA_CHECK(cudaEventElapsedTime(&iterMsEv, evI0, evI1));
        CUDA_CHECK(cudaEventDestroy(evI0));
        CUDA_CHECK(cudaEventDestroy(evI1));
      #ifndef CUDA_ARCH
        iterMs = static_cast<double>(iterMsEv);
      #endif
    }

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
    float shadeMsEv = 0.0f; cudaEvent_t evS0 = nullptr, evS1 = nullptr;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&evS0));
        CUDA_CHECK(cudaEventCreate(&evS1));
        CUDA_CHECK(cudaEventRecord(evS0, 0));
    }

    {
        dim3 block(32, 8);
        dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
        shade_from_iterations<<<grid, block>>>(
            devSurface,
            static_cast<const int*>(d_iterations.get()),
            width, height, maxIterations
        );
        CUDA_CHECK(cudaGetLastError());
    }

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
    entMs = ecMs * 0.5;  // grobe Aufteilung für Log-Zwecke
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

    // Timings an RendererState (GPU-Anteil = Iter + Shade)
    state.lastTimings.valid            = true;
#ifndef CUDA_ARCH
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = iterMs + shadeMs;
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
        LUCHS_LOG_HOST("[PERF] path=nm mp=%.2f iterGPU=%.2f shade=%.2f en=%.2f ct=%.2f tt=%.2f",
                       mapMs, iterMs, shadeMs, entMs, conMs, totalMs);
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
