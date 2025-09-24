///// Otter: OpenGL PBO interop; map/unmap + pointer retrieval logged deterministically.
///// Schneefuchs: No GL forward-decls; use GLEW properly; numeric CUDA rc codes; device-log flush on errors.
///// Maus: One implementation path (Capybara); static CUDA events with lazy init; pause toggle handled centrally.
///// Datei: src/cuda_interop.cu

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "frame_context.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"
#include "colorize_iterations.cuh"
#include "capybara_frame_pipeline.cuh"

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>   // std::max, std::abs

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace {
    // ---- CUDA timing events (lazy) ---------------------------------------
    static cudaEvent_t s_evStart = nullptr;
    static cudaEvent_t s_evStop  = nullptr;

    inline void ensureEventsOnce() {
        static bool done = false;
        if (done) return;
        done = true;
        cudaError_t rc = cudaEventCreate(&s_evStart); // timing enabled
        if (rc != cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA][ERR] eventCreate start rc=%d", (int)rc);
            LuchsLogger::flushDeviceLogToHost(0);
            throw std::runtime_error("cudaEventCreate(start) failed");
        }
        rc = cudaEventCreate(&s_evStop);
        if (rc != cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA][ERR] eventCreate stop rc=%d", (int)rc);
            LuchsLogger::flushDeviceLogToHost(0);
            throw std::runtime_error("cudaEventCreate(stop) failed");
        }
    }

    // ---- PBO CUDA resources ----------------------------------------------
    static std::vector<CudaInterop::bear_CudaPBOResource> s_pboResources;
    static bool s_pboActive = false;

    // ---- Global pause flag for zoom logic --------------------------------
    static bool s_pauseZoom = false;

    // RAII-Guard: map beim Bau, unmap beim Zerstören
    struct MapGuard {
        CudaInterop::bear_CudaPBOResource* res = nullptr;
        void*   ptr   = nullptr;
        size_t  bytes = 0;
        explicit MapGuard(CudaInterop::bear_CudaPBOResource* r) : res(r) {
            if (res) ptr = res->mapAndLog(bytes);
        }
        ~MapGuard() {
            if (res) res->unmapAndLog();
        }
    };

    // Helpers for safe attribute read
    static int getAttrSafe(cudaDeviceAttr attr, int dev) {
        int v = 0;
        auto e = cudaDeviceGetAttribute(&v, attr, dev);
        if (e != cudaSuccess) return -1;
        return v;
    }

    static inline void throw_with_log(const char* msg, cudaError_t rc) {
        LUCHS_LOG_HOST("[CUDA][ERR] %s rc=%d", msg ? msg : "(null)", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        throw std::runtime_error(msg ? msg : "CUDA error");
    }
} // anon

namespace CudaInterop {

bool precheckCudaRuntime() noexcept {
    int n = 0;
    cudaError_t rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess || n <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CUDA][PRECHECK] deviceCount rc=%d count=%d", (int)rc, n);
        }
        return false;
    }
    return true;
}

void setPauseZoom(bool paused) noexcept { s_pauseZoom = paused; }
bool getPauseZoom() noexcept { return s_pauseZoom; }

void registerAllPBOs(const unsigned int* pboIds, int count) {
    s_pboResources.clear();
    s_pboResources.reserve((size_t)count);
    for (int i = 0; i < count; ++i) {
        s_pboResources.emplace_back((GLuint)pboIds[i]);
    }
    s_pboActive = (count > 0);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] registered %d CUDA resources", count);
    }
}

void unregisterAllPBOs() noexcept {
    s_pboResources.clear(); // dtors unmap+unregister
    s_pboActive = false;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] unregistered all CUDA resources");
    }
}

void logCudaDeviceContext(const char* tag) noexcept {
    int rt = -1, drv = -1, dev = -1;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaGetDevice(&dev);
    char name[256] = {0};
    int ccM = -1, ccN = -1, mp = -1, smpb = -1;
    if (dev >= 0) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, dev);
        std::strncpy(name, p.name, sizeof(name)-1);
        ccM = getAttrSafe(cudaDevAttrComputeCapabilityMajor, dev);
        ccN = getAttrSafe(cudaDevAttrComputeCapabilityMinor, dev);
        mp  = getAttrSafe(cudaDevAttrMultiProcessorCount, dev);
        smpb= getAttrSafe(cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    }
    LUCHS_LOG_HOST("[CUDA][CTX][%s] rt=%d drv=%d dev=%d name=\"%s\" cc=%d.%d MP=%d thr/MP=%d",
                   (tag?tag:"-"), rt, drv, dev, name, ccM, ccN, mp, smpb);
}

// ----------------------------------------------------------------------------------
// Hauptpfad: Capybara render -> colorize to PBO (ohne Host-Sync; optional Perf-Sync)
// ----------------------------------------------------------------------------------
void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    int   width,
    int   height,
    float zoom,
    float offsetX,
    float offsetY,
    int   maxIterations,
    float& newOffsetX,
    float& newOffsetY,
    bool&  shouldZoom,
    RendererState& state,
    cudaStream_t renderStream
){
    (void)zoom;           // /WX: avoid unused in NVCC host pass; zoom handled upstream
    (void)newOffsetX;     // zoom/offset update handled upstream
    (void)newOffsetY;
    (void)shouldZoom;

    if (!s_pboActive) {
        LUCHS_LOG_HOST("[PBO][ERR] render called without registered PBOs");
        // WICHTIG: Kein Device-Log-Flush hier (Logger evtl. noch nicht init)
        state.skipUploadThisFrame = true;
        return;
    }
    if (width <= 0 || height <= 0) {
        LUCHS_LOG_HOST("[CUDA][ERR] invalid framebuffer dims %dx%d", width, height);
        state.skipUploadThisFrame = true;
        return;
    }

    ensureEventsOnce();
    (void)cudaGetLastError(); // clear sticky

    // ---- 1) Map current PBO slot ----------------------------------------
    const size_t needBytes = size_t(width) * size_t(height) * sizeof(uchar4);
    const int ix = (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) ? state.pboIndex : 0;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][MAP] try ring=%d need=%zu", ix, (size_t)needBytes);
    }

    MapGuard map(&s_pboResources[ix]);

    // deterministische Fehlerpfade: loggen + Frame degradieren, NICHT werfen
    if (!map.ptr) {
        cudaError_t rcMap = cudaGetLastError(); // evtl. von Map
        LUCHS_LOG_HOST("[PBO][MAP][ERR] null ptr ring=%d need=%zu rc=%d", ix, (size_t)needBytes, (int)rcMap);
        LuchsLogger::flushDeviceLogToHost(0);
        state.skipUploadThisFrame = true;
        return; // Frame ohne Upload beenden
    }
    if (map.bytes < needBytes) {
        LUCHS_LOG_HOST("[PBO][MAP][ERR] size mismatch ring=%d got=%zu need=%zu", ix, (size_t)map.bytes, (size_t)needBytes);
        LuchsLogger::flushDeviceLogToHost(0);
        state.skipUploadThisFrame = true;
        return;
    }

    if (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) {
        state.ringUse[state.pboIndex]++;
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][MAP] ok ring=%d ptr=%p bytes=%zu", ix, map.ptr, (size_t)map.bytes);
    }

    // ---- 2) Capybara render (iterations) --------------------------------
    const double cx   = (double)offsetX;
    const double cy   = (double)offsetY;

    // Quadratische Pixel erzwingen: gleiche Schrittweite in X/Y,
    // Vorzeichen aus dem GL-PixelScale übernehmen (Y ist oft negativ).
    const double sx = (double)state.pixelScale.x;
    const double sy = (double)state.pixelScale.y;

    double step = std::max(std::abs(sx), std::abs(sy));
    if (!(step > 0.0)) {
        const double baseSpan = 3.5;
        step = baseSpan / (std::max(1, width) * std::max(1.0f, zoom));
    }
    const double stepX = std::copysign(step, (sx == 0.0 ? 1.0 : sx));
    const double stepY = std::copysign(step, (sy == 0.0 ? -1.0 : sy));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CAPY][ARGS] cx=%.9f cy=%.9f stepX=%.11f stepY=%.11f it=%d w=%d h=%d",
                       cx, cy, stepX, stepY, maxIterations, width, height);
    }

    cudaError_t rc = cudaEventRecord(s_evStart, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(start) before capy_render", rc);

    capy_render(
        static_cast<uint16_t*>(d_iterations.get()),
        width, height,
        cx, cy,
        stepX, stepY,
        maxIterations,
        renderStream,
        state.evEcDone // optional event reuse
    );

    // Launch-Fehler peek (numerisch loggen)
    rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("capy_render launch", rc);

    rc = cudaEventRecord(s_evStop, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after capy_render", rc);

    if constexpr (Settings::performanceLogging) {
        cudaError_t rcSync = cudaEventSynchronize(s_evStop);
        if (rcSync != cudaSuccess) throw_with_log("capy_render sync", rcSync);
    }

    // ---- 3) Colorize into mapped PBO ------------------------------------
    rc = cudaEventRecord(s_evStart, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(start) before colorize", rc);

    colorize_iterations_to_pbo(
        static_cast<const uint16_t*>(d_iterations.get()),
        static_cast<uchar4*>(map.ptr),
        width, height, maxIterations,
        renderStream
    );

    rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("colorize launch", rc);

    rc = cudaEventRecord(s_evStop, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after colorize", rc);

    if constexpr (Settings::performanceLogging) {
        cudaError_t rcSync = cudaEventSynchronize(s_evStop);
        if (rcSync != cudaSuccess) throw_with_log("colorize sync", rcSync);
    }

    // ---- 4) done; MapGuard dtor will unmap -------------------------------
}

void renderCudaFrame(RendererState& state, const FrameContext& fctx, float& newOffsetX, float& newOffsetY) {
    float offx = fctx.offset.x;
    float offy = fctx.offset.y;
    bool  shouldZoom = false;

    renderCudaFrame(
        state.d_iterations,
        fctx.width, fctx.height,
        fctx.zoom,
        offx, offy,
        fctx.maxIterations,
        newOffsetX, newOffsetY,
        shouldZoom,
        state,
        state.renderStream
    );
}

// --- ABI-Kompat: alter Name bleibt als Alias verfügbar -----------------
void logCudaContext(const char* tag) noexcept {
    logCudaDeviceContext(tag);
}

} // namespace CudaInterop
