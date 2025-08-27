// MAUS:
// Datei: src/cuda_interop.cu
// 🐜 Schwarze Ameise: Klare Parametrisierung, deterministisches Logging, robustes Ressourcenhandling.
// 🦦 Otter: Explizite und einheitliche Übergabe aller Parameter. Fehler- und Kontextlogging überall. (Bezug zu Otter)
// 🦊 Schneefuchs: Keine impliziten Zugriffe, transparente Speicher- und Fehlerprüfung. (Bezug zu Schneefuchs)

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <vector>
#include <stdexcept>

#ifndef CUDA_ARCH
#include <chrono>
#endif

namespace CudaInterop {

// 🐑 Schneefuchs: TU-lokaler Zustand – kein Header-Touch.
static bear_CudaPBOResource* pboResource      = nullptr;
static bool pauseZoom                         = false;
static bool luchsBabyInitDone                 = false;
static bool s_deviceInitDone                  = false;
static int  s_frameCounter                    = 0;

// 🐑 Schneefuchs: Pinned-Host-Registrierung cachen (schnellere D2H).
static void*  s_hostRegEntropyPtr  = nullptr;
static size_t s_hostRegEntropyBytes= 0;
static void*  s_hostRegContrastPtr = nullptr;
static size_t s_hostRegContrastBytes=0;

// 🐑 Schneefuchs: Einmaliges Device-Set, deterministisch.
static inline void ensureDeviceOnce() {
    if (!s_deviceInitDone) {
        CUDA_CHECK(cudaSetDevice(0));
        s_deviceInitDone = true;
    }
}

// 🐑 Schneefuchs: Hostspeicher (Vector-Backing) page-locken, nur bei Realloc/Capacity-Change.
static inline void ensureHostPinned(std::vector<float>& vec, void*& regPtr, size_t& regBytes) {
    const size_t cap = vec.capacity();
    if (cap == 0) { // Nichts zu pinnen
        if (regPtr) { CUDA_CHECK(cudaHostUnregister(regPtr)); regPtr = nullptr; regBytes = 0; }
        return;
    }
    void* ptr = static_cast<void*>(vec.data());
    const size_t bytes = cap * sizeof(float);
    if (ptr != regPtr || bytes != regBytes) {
        if (regPtr) CUDA_CHECK(cudaHostUnregister(regPtr));
        // Portable für GL/CUDA Mischbetrieb
        CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
        regPtr  = ptr;
        regBytes= bytes;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] host-register ptr=%p bytes=%zu", ptr, bytes);
        }
    }
}

void logCudaDeviceContext(const char* context) {
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CTX] %s: cudaGetDevice() = %d (%s)", context, device, cudaGetErrorString(err));
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

    GLint boundBefore = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBefore);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CHECK] GL bind state BEFORE bind: %d", boundBefore);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.id());
    GLint boundAfter = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundAfter);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CHECK] GL bind state AFTER bind: %d (expected: %u)", boundAfter, pbo.id());
    }

    if (boundAfter != static_cast<GLint>(pbo.id())) {
        LUCHS_LOG_HOST("[FATAL] GL bind failed - buffer %u was not bound (GL reports: %d)", pbo.id(), boundAfter);
        throw std::runtime_error("glBindBuffer(GL_PIXEL_UNPACK_BUFFER) failed");
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CU-PBO] Preparing to register PBO ID %u", pbo.id());
    }

    pboResource = new bear_CudaPBOResource(pbo.id());

    if constexpr (Settings::debugLogging) {
        logCudaDeviceContext("after registerPBO");
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
    ++s_frameCounter;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ENTER] renderCudaFrame(tileSize=%d)", tileSize);
        logCudaDeviceContext("renderCudaFrame ENTER");
    }

    if (!pboResource)
        throw std::runtime_error("[FATAL] CUDA PBO not registered!");

#ifndef CUDA_ARCH
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs = 0.0, mandelbrotMs = 0.0, entMs = 0.0, conMs = 0.0, flushMs = 0.0;
#endif

    const int totalPixels = width * height;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    // --- Size sanity & allocation guards (Ameise) ---
    const size_t it_bytes       = static_cast<size_t>(totalPixels) * sizeof(int);
    const size_t entropy_bytes  = static_cast<size_t>(numTiles)    * sizeof(float);
    const size_t contrast_bytes = static_cast<size_t>(numTiles)    * sizeof(float);

    const size_t d_it_size       = d_iterations.size();
    const size_t d_entropy_size  = d_entropy.size();
    const size_t d_contrast_size = d_contrast.size();

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[SANITY] w=%d h=%d pixels=%d tileSize=%d tiles=%d (%d x %d)",
                       width, height, totalPixels, tileSize, numTiles, tilesX, tilesY);
        LUCHS_LOG_HOST("[SANITY] alloc(it=%zu, entropy=%zu, contrast=%zu) need(it=%zu, entropy=%zu, contrast=%zu)",
                       d_it_size, d_entropy_size, d_contrast_size, it_bytes, entropy_bytes, contrast_bytes);
    }

    bool alloc_ok = true;
    if (d_it_size < it_bytes) {
        LUCHS_LOG_HOST("[FATAL] iterations buffer too small: have=%zu need=%zu", d_it_size, it_bytes);
        alloc_ok = false;
    }
    if (d_entropy_size < entropy_bytes) {
        LUCHS_LOG_HOST("[FATAL] entropy buffer too small: have=%zu need=%zu (tiles=%d)", d_entropy_size, entropy_bytes, numTiles);
        alloc_ok = false;
    }
    if (d_contrast_size < contrast_bytes) {
        LUCHS_LOG_HOST("[FATAL] contrast buffer too small: have=%zu need=%zu (tiles=%d)", d_contrast_size, contrast_bytes, numTiles);
        alloc_ok = false;
    }
    if (!alloc_ok) {
        throw std::runtime_error("CudaInterop::renderCudaFrame: device buffers undersized for current tile layout");
    }

    // 🐑 Schneefuchs: Keine globale Synchronisation vor MAP – Reihenfolge genügt.
    // (Alle bisherigen Ops betreffen keine GL-Ressource.)

    // Device-Seitige Clears (deterministisch), optional loggen
    CUDA_CHECK(cudaMemset(d_iterations.get(), 0, d_iterations.size()));
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[MEM] d_iterations memset: %d pixels -> %zu bytes", totalPixels, d_iterations.size());
    }
    CUDA_CHECK(cudaMemset(d_entropy.get(),   0, d_entropy.size()));
    CUDA_CHECK(cudaMemset(d_contrast.get(),  0, d_contrast.size()));

    // Map PBO (Host-Zeit messen)
#ifndef CUDA_ARCH
    const auto tMap0 = std::chrono::high_resolution_clock::now();
#endif
    size_t sizeBytes = 0;
    uchar4* devPtr = static_cast<uchar4*>(pboResource->mapAndLog(sizeBytes));
#ifndef CUDA_ARCH
    const auto tMap1 = std::chrono::high_resolution_clock::now();
    mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
#endif

    if (!devPtr) {
        LUCHS_LOG_HOST("[FATAL] Kernel skipped: surface pointer is null");
        return;
    }

    // Sanity-Check auf PBO-Größe
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar4);
    if (sizeBytes < expected) {
        LUCHS_LOG_HOST("[FATAL] PBO size too small: got=%zu need=%zu (w=%d h=%d)", sizeBytes, expected, width, height);
        pboResource->unmap();
        throw std::runtime_error("PBO byte size mismatch");
    }

    if (!luchsBabyInitDone) {
        LuchsLogger::initCudaLogBuffer(0);
        luchsBabyInitDone = true;
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST(
            "[KERNEL] launch_mandelbrotHybrid(surface=%p, w=%d, h=%d, zoom=%.5f, offset=(%.5f,%.5f), iter=%d, tile=%d)",
            (void*)devPtr, width, height, zoom, offset.x, offset.y, maxIterations, tileSize
        );
    }

    // 🐑 Schneefuchs: Kernel-Zeit via Events (nur wenn Logging aktiv).
    float mandelbrotMsEv = 0.0f;
    cudaEvent_t evK0 = nullptr, evK1 = nullptr;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventCreate(&evK0));
        CUDA_CHECK(cudaEventCreate(&evK1));
        CUDA_CHECK(cudaEventRecord(evK0, 0)); // default stream
    }

    // Launch
    launch_mandelbrotHybrid(
        devPtr,
        static_cast<int*>(d_iterations.get()),
        width, height, zoom, offset, maxIterations, tileSize
    );

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        CUDA_CHECK(cudaEventRecord(evK1, 0));
        CUDA_CHECK(cudaEventSynchronize(evK1));
        CUDA_CHECK(cudaEventElapsedTime(&mandelbrotMsEv, evK0, evK1));
        mandelbrotMs = (double)mandelbrotMsEv;
        CUDA_CHECK(cudaEventDestroy(evK0));
        CUDA_CHECK(cudaEventDestroy(evK1));
    }

    // Optional: kleines It-Sample zum Sanity-Check
    if constexpr (Settings::debugLogging) {
        int dbg_after[3] = {};
        CUDA_CHECK(cudaMemcpy(dbg_after, d_iterations.get(), sizeof(dbg_after), cudaMemcpyDeviceToHost));
        LUCHS_LOG_HOST("[KERNEL] iters sample: %d %d %d", dbg_after[0], dbg_after[1], dbg_after[2]);
    }

    // Entropy/Contrast (Host-Timing als Approx.)
#ifndef CUDA_ARCH
    const auto tEnt0 = std::chrono::high_resolution_clock::now();
#endif
    ::computeCudaEntropyContrast(
        static_cast<const int*>(d_iterations.get()),
        static_cast<float*>(d_entropy.get()),
        static_cast<float*>(d_contrast.get()),
        width, height, tileSize, maxIterations
    );
#ifndef CUDA_ARCH
    const auto tEnt1 = std::chrono::high_resolution_clock::now();
    const double ecMs = std::chrono::duration<double, std::milli>(tEnt1 - tEnt0).count();
    // Aufteilen (beste Schätzung, ohne innere Events)
    entMs = ecMs * 0.5;
    conMs = ecMs * 0.5;
#endif

    // Host-Ziele vorbereiten (keine Reallocs, dann pinnen)
    if ((size_t)h_entropy.capacity() < (size_t)numTiles) h_entropy.reserve((size_t)numTiles);
    if ((size_t)h_contrast.capacity() < (size_t)numTiles) h_contrast.reserve((size_t)numTiles);
    ensureHostPinned(h_entropy,  s_hostRegEntropyPtr,  s_hostRegEntropyBytes);
    ensureHostPinned(h_contrast, s_hostRegContrastPtr, s_hostRegContrastBytes);

    h_entropy.resize((size_t)numTiles);
    h_contrast.resize((size_t)numTiles);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[COPY] entropy D->H: dst=%p src=%p bytes=%zu",
                       (void*)h_entropy.data(), d_entropy.get(), entropy_bytes);
        LUCHS_LOG_HOST("[COPY] contrast D->H: dst=%p src=%p bytes=%zu",
                       (void*)h_contrast.data(), d_contrast.get(), contrast_bytes);
    }

    CUDA_CHECK(cudaMemcpy(h_entropy.data(),  d_entropy.get(),  entropy_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast.get(), contrast_bytes, cudaMemcpyDeviceToHost));

    // Zoom-Kommunikation
    shouldZoom = false;
    newOffset  = offset;

    // Device-Logs nicht immer spülen – nur bei Fehler oder Modulo.
    cudaError_t lastErr = cudaPeekAtLastError();
#ifndef CUDA_ARCH
    const auto tFlush0 = std::chrono::high_resolution_clock::now();
#endif
    if (lastErr != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA] Error detected: code=%d -> flushing device log", (int)lastErr);
        LuchsLogger::flushDeviceLogToHost();
#ifndef CUDA_ARCH
        const auto tFlush1 = std::chrono::high_resolution_clock::now();
        flushMs = std::chrono::duration<double, std::milli>(tFlush1 - tFlush0).count();
#endif
    } else if ((s_frameCounter % 30) == 0 || (Settings::debugLogging && (s_frameCounter % 5) == 0)) {
        LuchsLogger::flushDeviceLogToHost();
#ifndef CUDA_ARCH
        const auto tFlush1 = std::chrono::high_resolution_clock::now();
        flushMs = std::chrono::duration<double, std::milli>(tFlush1 - tFlush0).count();
#endif
    }

    pboResource->unmap();

#ifndef CUDA_ARCH
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    // 🐑 Schneefuchs: Timings an RendererState zurückgeben, wenn Feld vorhanden
    state.lastTimings.valid            = true;
#ifndef CUDA_ARCH
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = mandelbrotMs;
    state.lastTimings.mandelbrotLaunch = 0.0;   // Host-Launch vernachlässigbar; optional später befüllen
    state.lastTimings.mandelbrotSync   = 0.0;   // durch Events erfasst → 0
    state.lastTimings.entropy          = entMs;
    state.lastTimings.contrast         = conMs;
    state.lastTimings.deviceLogFlush   = flushMs;
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
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PERF] mp=%.2f md=%.2f en=%.2f ct=%.2f fl=%.2f tt=%.2f",
               mapMs,    mandelbrotMs, entMs,   conMs,   flushMs, totalMs);
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

} // namespace CudaInterop
