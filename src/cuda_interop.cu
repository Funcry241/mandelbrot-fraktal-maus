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

    // ---- Debug: Fallback-Drawer in den PBO (zeigt Upload-/Blit-Pfad an) ---
    __global__ void dbg_fill_checker(uchar4* rgba, int w, int h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;
        int idx = y * w + x;
        const int s = 32;
        bool chk = ((x / s) ^ (y / s)) & 1;
        unsigned char r = (unsigned char)((x * 255) / max(1, w - 1));
        unsigned char g = (unsigned char)((y * 255) / max(1, h - 1));
        unsigned char b = chk ? 255 : 64;
        rgba[idx] = make_uchar4(r, g, b, 255);
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

    if (!map.ptr) {
        cudaError_t rcMap = cudaGetLastError(); // evtl. von Map
        LUCHS_LOG_HOST("[PBO][MAP][ERR] null ptr ring=%d need=%zu rc=%d", ix, (size_t)needBytes, (int)rcMap);
        LuchsLogger::flushDeviceLogToHost(0);
        state.skipUploadThisFrame = true;
        return;
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
    const double cx = (double)offsetX;
    const double cy = (double)offsetY;

    // Schrittweiten direkt aus PixelScale
    double stepX = (double)state.pixelScale.x;
    double stepY = (double)state.pixelScale.y;
    if (stepY > 0.0) stepY = -stepY; // Bildschirm-Y nach unten

    if (!std::isfinite(stepX) || !std::isfinite(stepY) || stepX == 0.0 || stepY == 0.0) {
        stepX = 3.5 / std::max(1, width);
        stepY = -2.0 / std::max(1, height);
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CAPY][ARGS] cx=%.9f cy=%.9f stepX=%.9g stepY=%.9g it=%d w=%d h=%d",
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

    rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("capy_render launch", rc);

    rc = cudaEventRecord(s_evStop, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after capy_render", rc);

    if constexpr (Settings::performanceLogging) {
        cudaError_t rcSync = cudaEventSynchronize(s_evStop);
        if (rcSync != cudaSuccess) throw_with_log("capy_render sync", rcSync);
    }

    // ---- 2.5) Debug: Iteration-Stichprobe & optional Sanity-Fallback -----
    bool useDbgPattern = false;
    if constexpr (Settings::debugLogging) {
        const int cxp = width  / 2, cyp = height / 2;
        int xs[4] = { std::max(0, cxp - width/4), std::min(width-1, cxp + width/4), cxp, cxp };
        int ys[4] = { cyp, cyp, std::max(0, cyp - height/4), std::min(height-1, cyp + height/4) };
        uint16_t hostIt[4] = {0,0,0,0};
        for (int i = 0; i < 4; ++i) {
            size_t idx = (size_t)ys[i] * (size_t)width + (size_t)xs[i];
            CUDA_CHECK(cudaMemcpyAsync(&hostIt[i],
                                       static_cast<const uint16_t*>(d_iterations.get()) + idx,
                                       sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost,
                                       renderStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(renderStream));
        LUCHS_LOG_HOST("[CAPY][SAMPLE] it={%u,%u,%u,%u}", (unsigned)hostIt[0], (unsigned)hostIt[1],
                       (unsigned)hostIt[2], (unsigned)hostIt[3]);

        if (hostIt[0]==0 && hostIt[1]==0 && hostIt[2]==0 && hostIt[3]==0) {
            // Wenn alles 0 ist, zeige Pattern (prüft Upload-/Blit-Pfad)
            useDbgPattern = true;
        }
    }

    // ---- 3) Colorize (oder Debug-Pattern) --------------------------------
    rc = cudaEventRecord(s_evStart, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(start) before colorize/dbg", rc);

    if (useDbgPattern) {
        dim3 block(32, 8);
        dim3 grid((unsigned)((width  + block.x - 1) / block.x),
                  (unsigned)((height + block.y - 1) / block.y));
        dbg_fill_checker<<<grid, block, 0, renderStream>>>(static_cast<uchar4*>(map.ptr), width, height);
        rc = cudaPeekAtLastError();
        if (rc != cudaSuccess) throw_with_log("dbg_fill_checker launch", rc);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[DBG] drew checkerboard into PBO (skipped colorizer this frame)");
        }
    } else {
        colorize_iterations_to_pbo(
            static_cast<const uint16_t*>(d_iterations.get()),
            static_cast<uchar4*>(map.ptr),
            width, height, maxIterations,
            renderStream
        );
        rc = cudaPeekAtLastError();
        if (rc != cudaSuccess) throw_with_log("colorize launch", rc);
    }

    rc = cudaEventRecord(s_evStop, renderStream);
    if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after colorize/dbg", rc);

    if constexpr (Settings::performanceLogging) {
        cudaError_t rcSync = cudaEventSynchronize(s_evStop);
        if (rcSync != cudaSuccess) throw_with_log("colorize/dbg sync", rcSync);
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
