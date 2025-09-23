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
        CUDA_CHECK(cudaEventCreate(&s_evStart)); // timing enabled
        CUDA_CHECK(cudaEventCreate(&s_evStop));
    }

    // ---- PBO CUDA resources ----------------------------------------------
    static std::vector<CudaInterop::bear_CudaPBOResource> s_pboResources;
    static bool s_pboActive = false;

    // ---- Global pause flag for zoom logic --------------------------------
    static bool s_pauseZoom = false;

    // Guard to map the current ring entry and unmap at scope end
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

void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
    int   width,
    int   height,
    float zoom,
    float offsetX,
    float offsetY,
    int   maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float& newOffsetX,
    float& newOffsetY,
    bool&  shouldZoom,
    int    tileSize,
    RendererState& state,
    cudaStream_t renderStream,
    cudaStream_t copyStream
){
    (void)d_entropy; (void)d_contrast;
    (void)h_entropy; (void)h_contrast;
    (void)copyStream;
    (void)newOffsetX; (void)newOffsetY; (void)shouldZoom; // zoom handled upstream
    (void)zoom; // /WX: avoid C4100 in NVCC host pass

    if (!s_pboActive) throw std::runtime_error("[FATAL] CUDA PBO not registered!");
    if (width <= 0 || height <= 0) throw std::runtime_error("invalid framebuffer dims");
    if (tileSize <= 0) {
        int was = tileSize; tileSize = Settings::BASE_TILE_SIZE;
        LUCHS_LOG_HOST("[WARN] tileSize<=0 (%d) -> using %d", was, tileSize);
    }

    ensureEventsOnce();
    (void)cudaGetLastError(); // clear sticky

    // ---- 1) Map current PBO slot ----------------------------------------
    const size_t needBytes = size_t(width) * size_t(height) * sizeof(uchar4);
    const int ix = (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) ? state.pboIndex : 0;
    MapGuard map(&s_pboResources[ix]);
    if (!map.ptr) throw std::runtime_error("pboResource->map() returned null");
    if (map.bytes < needBytes) throw std::runtime_error("PBO byte size mismatch");
    if (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) {
        state.ringUse[state.pboIndex]++;
    }

    // ---- 2) Capybara render (iterations) --------------------------------
    const double cx   = (double)offsetX;
    const double cy   = (double)offsetY;
    const double step = std::max(std::abs(state.pixelScale.x), std::abs(state.pixelScale.y));
    CUDA_CHECK(cudaEventRecord(s_evStart, renderStream));
    capy_render_and_analyze(
        static_cast<uint16_t*>(d_iterations.get()),
        nullptr, nullptr, // EC removed
        width, height,
        cx, cy,
        step, step,
        maxIterations,
        /*tileSize*/   tileSize,   // kept for ABI parity
        renderStream,
        state.evEcDone,
        Settings::capybaraEnabled
    );
    CUDA_CHECK(cudaEventRecord(s_evStop, renderStream));
    if constexpr (Settings::performanceLogging) {
        cudaError_t rcCapySync = cudaEventSynchronize(s_evStop);
        if (rcCapySync != cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA][ERR] capybara sync rc=%d", (int)rcCapySync);
            LuchsLogger::flushDeviceLogToHost(0);
            throw std::runtime_error("CUDA failure: capy_render_and_analyze");
        }
    }

    // ---- 3) Colorize into mapped PBO ------------------------------------
    CUDA_CHECK(cudaEventRecord(s_evStart, renderStream));
    colorize_iterations_to_pbo(
        static_cast<const uint16_t*>(d_iterations.get()),
        static_cast<uchar4*>(map.ptr),
        width, height, maxIterations,
        renderStream
    );
    CUDA_CHECK(cudaEventRecord(s_evStop, renderStream));
    if constexpr (Settings::performanceLogging) {
        cudaError_t rcColSync = cudaEventSynchronize(s_evStop);
        if (rcColSync != cudaSuccess) {
            LUCHS_LOG_HOST("[CUDA][ERR] colorize sync rc=%d", (int)rcColSync);
            LuchsLogger::flushDeviceLogToHost(0);
            throw std::runtime_error("CUDA failure: colorize_iterations_to_pbo");
        }
    }

    // ---- 4) done; MapGuard dtor will unmap --------------------------------
    (void)renderStream;
}

void renderCudaFrame(RendererState& state, const FrameContext& fctx, float& newOffsetX, float& newOffsetY) {
    float offx = fctx.offset.x;
    float offy = fctx.offset.y;
    bool  shouldZoom = false;
    renderCudaFrame(
        state.d_iterations,
        state.d_entropy,
        state.d_contrast,
        fctx.width, fctx.height,
        fctx.zoom,
        offx, offy,
        fctx.maxIterations,
        /*h_entropy*/ const_cast<std::vector<float>&>(state.h_entropy),
        /*h_contrast*/const_cast<std::vector<float>&>(state.h_contrast),
        newOffsetX, newOffsetY,
        shouldZoom,
        fctx.tileSize,
        state,
        state.renderStream,
        state.copyStream
    );
}

// --- ABI-Kompat: alter Name bleibt als Alias verfügbar -----------------
void logCudaContext(const char* tag) noexcept {
    logCudaDeviceContext(tag);
}

} // namespace CudaInterop
