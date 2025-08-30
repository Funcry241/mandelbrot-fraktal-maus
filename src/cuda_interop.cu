// Datei: src/cuda_interop.cu
// 🐜 Schwarze Ameise: Klare Parametrisierung, deterministisches Logging, robustes Ressourcenhandling.
// 🦦 Otter: Sichtbarer Test-Pattern-Path (GPU-only) für schnelle Verifikation; Iterations-/Analyse-Pfade später einschaltbar.
// 🦊 Schneefuchs: Saubere Fehlerbehandlung, Null Seiteneffekte außerhalb der TU; /WX-fest.

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#ifndef CUDA_ARCH
  #include <chrono>
#endif

namespace CudaInterop {

// ------------------------------ TU-lokaler Zustand ---------------------------
static bear_CudaPBOResource* s_pbo      = nullptr;
static bool                  s_devReady = false;
static bool                  s_pauseZoom= false;

// Host-Pinning (für spätere schnelle D2H von Analysepuffern)
static void*  s_hostRegEntropyPtr    = nullptr;
static size_t s_hostRegEntropyBytes  = 0;
static void*  s_hostRegContrastPtr   = nullptr;
static size_t s_hostRegContrastBytes = 0;

static inline void ensureDeviceOnce() {
    if (!s_devReady) {
        CUDA_CHECK(cudaSetDevice(0));
        s_devReady = true;
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

// ------------------------------ Sichtbarer Test-Pattern-Kernel ---------------
// Ein einfacher Farbverlauf + kariertes Blau, damit man SOFORT etwas sieht.
__global__ void fill_test_pattern(uchar4* surf, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const unsigned char r = static_cast<unsigned char>((x * 255) / max(w, 1));
    const unsigned char g = static_cast<unsigned char>((y * 255) / max(h, 1));
    const unsigned char b = (((x >> 4) ^ (y >> 4)) & 1) ? 255 : 0; // grobes Schachbrett
    surf[y * w + x] = make_uchar4(r, g, b, 255);
}

// ------------------------------ PBO-Registration -----------------------------
void registerPBO(const Hermelin::GLBuffer& pbo) {
    if (s_pbo) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[WARN] registerPBO: already registered (id=%u)", pbo.id());
        }
        return;
    }

    ensureDeviceOnce();

    // Sanity: war der Bind erfolgreich?
    GLint boundBefore = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBefore);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.id());
    GLint boundAfter = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundAfter);

    if (boundAfter != static_cast<GLint>(pbo.id())) {
        LUCHS_LOG_HOST("[FATAL] GL bind failed - buffer %u was not bound (GL reports: %d)", pbo.id(), boundAfter);
        throw std::runtime_error("glBindBuffer(GL_PIXEL_UNPACK_BUFFER) failed");
    }

    s_pbo = new bear_CudaPBOResource(pbo.id());

    // Warm-up (einmaliges Map/Unmap)
    size_t warmBytes = 0;
    if (auto* ptr = s_pbo->mapAndLog(warmBytes)) {
        (void)ptr;
        s_pbo->unmap();
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PBO] warm-up map/unmap done (%zu bytes)", warmBytes);
        }
    }

    // Ursprünglichen Bind wiederherstellen
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(boundBefore));
}

// ------------------------------ Haupt-Frame-Render ---------------------------
void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,  // (derzeit ungenutzt im Testpattern)
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
    int width,
    int height,
    float /*zoom*/,
    float2 offset,
    int /*maxIterations*/,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize,
    RendererState& state
) {
#ifndef CUDA_ARCH
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs = 0.0, shadeMs = 0.0;
#endif

    (void)offset; (void)tileSize; // aktuell nicht benötigt

    if (!s_pbo)
        throw std::runtime_error("[FATAL] CUDA PBO not registered!");

    // PBO mappen -> CUDA-Surface bekommen
#ifndef CUDA_ARCH
    const auto tMap0 = std::chrono::high_resolution_clock::now();
#endif
    size_t surfBytes = 0;
    uchar4* devSurface = static_cast<uchar4*>(s_pbo->mapAndLog(surfBytes));
#ifndef CUDA_ARCH
    const auto tMap1 = std::chrono::high_resolution_clock::now();
    mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
#endif

    if (!devSurface) {
        LUCHS_LOG_HOST("[FATAL] surface pointer is null");
        throw std::runtime_error("pbo map returned null");
    }

    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar4);
    if (surfBytes < expected) {
        LUCHS_LOG_HOST("[FATAL] PBO size too small: got=%zu need=%zu (w=%d h=%d)", surfBytes, expected, width, height);
        s_pbo->unmap();
        throw std::runtime_error("PBO byte size mismatch");
    }

    // --- Sichtbarer Shader: Test-Pattern ---
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

#ifndef CUDA_ARCH
    cudaEvent_t evS0=nullptr, evS1=nullptr; float shadeMsEv=0.0f;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&evS0));
        CUDA_CHECK(cudaEventCreate(&evS1));
        CUDA_CHECK(cudaEventRecord(evS0, 0));
    }
#endif

    fill_test_pattern<<<grid, block>>>(devSurface, width, height);
    CUDA_CHECK(cudaGetLastError());

#ifndef CUDA_ARCH
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventRecord(evS1, 0));
        CUDA_CHECK(cudaEventSynchronize(evS1));
        CUDA_CHECK(cudaEventElapsedTime(&shadeMsEv, evS0, evS1));
        CUDA_CHECK(cudaEventDestroy(evS0));
        CUDA_CHECK(cudaEventDestroy(evS1));
        shadeMs = static_cast<double>(shadeMsEv);
    }
#endif

    // --- E/C für Heatmap vorbereiten (hier: Nullfelder, nur um Overlay zu beruhigen) ---
    const int tilesX = (width  + tileSize - 1) / max(tileSize, 1);
    const int tilesY = (height + tileSize - 1) / max(tileSize, 1);
    const size_t numTiles = static_cast<size_t>(tilesX) * static_cast<size_t>(tilesY);

    if (d_entropy.size() < numTiles * sizeof(float))  { d_entropy.resize(numTiles * sizeof(float)); }
    if (d_contrast.size() < numTiles * sizeof(float)) { d_contrast.resize(numTiles * sizeof(float)); }

    CUDA_CHECK(cudaMemset(d_entropy.get(),  0, d_entropy.size()));
    CUDA_CHECK(cudaMemset(d_contrast.get(), 0, d_contrast.size()));

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);

    ensureHostPinned(h_entropy,  s_hostRegEntropyPtr,  s_hostRegEntropyBytes);
    ensureHostPinned(h_contrast, s_hostRegContrastPtr, s_hostRegContrastBytes);

    CUDA_CHECK(cudaMemcpy(h_entropy.data(),  d_entropy.get(),  d_entropy.size(),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast.get(), d_contrast.size(), cudaMemcpyDeviceToHost));

    // Zoom-Flags fürs Host-System (vorerst aus)
    shouldZoom = false;
    newOffset  = make_float2(0.0f, 0.0f);

    // PBO wieder freigeben
    s_pbo->unmap();

#ifndef CUDA_ARCH
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    // Timings melden (GPU-Anteil = Pattern Shade)
    state.lastTimings.valid            = true;
#ifndef CUDA_ARCH
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = shadeMs;
    state.lastTimings.mandelbrotLaunch = 0.0;
    state.lastTimings.mandelbrotSync   = 0.0;
    state.lastTimings.entropy          = 0.0;
    state.lastTimings.contrast         = 0.0;
    state.lastTimings.deviceLogFlush   = 0.0;

    if constexpr (Settings::performanceLogging) {
        LUCHS_LOG_HOST("[PERF] path=test mp=%.2f shade=%.2f tt=%.2f", mapMs, shadeMs, totalMs);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[TIME] total=%.2f", totalMs);
    }
#else
    state.lastTimings.pboMap           = 0.0;
    state.lastTimings.mandelbrotTotal  = 0.0;
    state.lastTimings.mandelbrotLaunch = 0.0;
    state.lastTimings.mandelbrotSync   = 0.0;
    state.lastTimings.entropy          = 0.0;
    state.lastTimings.contrast         = 0.0;
    state.lastTimings.deviceLogFlush   = 0.0;
#endif
}

// ------------------------------ Hilfsfunktionen ------------------------------
void setPauseZoom(bool pause) { s_pauseZoom = pause; }
bool getPauseZoom()           { return s_pauseZoom; }

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
    // Host-Pins lösen
    if (s_hostRegEntropyPtr)  { cudaHostUnregister(s_hostRegEntropyPtr);  s_hostRegEntropyPtr  = nullptr; s_hostRegEntropyBytes  = 0; }
    if (s_hostRegContrastPtr) { cudaHostUnregister(s_hostRegContrastPtr); s_hostRegContrastPtr = nullptr; s_hostRegContrastBytes = 0; }

    delete s_pbo;
    s_pbo = nullptr;
}

void logCudaDeviceContext(const char* tag) {
    int dev = -1;
    cudaError_t e0 = cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaError_t e1 = (e0 == cudaSuccess && dev >= 0)
                   ? cudaGetDeviceProperties(&prop, dev)
                   : cudaErrorInvalidDevice;

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        if (e0 == cudaSuccess && e1 == cudaSuccess) {
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
