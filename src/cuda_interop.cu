///// Otter: OpenGL PBO interop; map/unmap + pointer retrieval logged deterministically.
///// Schneefuchs: No GL forward-decls; numeric CUDA rc codes; perf events only when enabled.
///// Maus: Single Capybara path; pause toggle centrally; heatmap metrics delegated.
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
#include "capybara_mapping.cuh"    // capy_pixel_steps_from_zoom_scale(...)
#include "heatmap_metrics.hpp"     // HeatmapMetrics::buildGPU

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace {

// ---- CUDA timing events (created only when performanceLogging) ----
static cudaEvent_t s_evStart = nullptr;
static cudaEvent_t s_evStop  = nullptr;

inline void ensureEventsOnce() {
    static bool done = false;
    if (done) return;
    done = true;
    auto rc = cudaEventCreate(&s_evStart);
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

// RAII-Guard: map on ctor, unmap on dtor
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
    auto rc = cudaGetDeviceCount(&n);
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
        snprintf(name, sizeof(name), "%s", p.name);
        ccM = getAttrSafe(cudaDevAttrComputeCapabilityMajor, dev);
        ccN = getAttrSafe(cudaDevAttrComputeCapabilityMinor, dev);
        mp  = getAttrSafe(cudaDevAttrMultiProcessorCount, dev);
        smpb= getAttrSafe(cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    }
    LUCHS_LOG_HOST("[CUDA][CTX][%s] rt=%d drv=%d dev=%d name=\"%s\" cc=%d.%d MP=%d thr/MP=%d",
                   (tag?tag:"-"), rt, drv, dev, name, ccM, ccN, mp, smpb);
}

// ------------------------------ main render path ------------------------------
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
    (void)newOffsetX; (void)newOffsetY; (void)shouldZoom;

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

    (void)cudaGetLastError(); // clear sticky

    // 1) map current PBO slot
    const size_t needBytes = size_t(width) * size_t(height) * sizeof(uchar4);
    const int ix = (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) ? state.pboIndex : 0;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][MAP] try ring=%d need=%zu", ix, (size_t)needBytes);
    }

    MapGuard map(&s_pboResources[ix]);

    if (!map.ptr) {
        const auto rcMap = cudaGetLastError();
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

    // 2) capybara render (iterations)
    const double cx = (double)offsetX;
    const double cy = (double)offsetY;

    const double sx = (double)state.pixelScale.x;
    const double sy = (double)state.pixelScale.y;
    double stepX = 0.0, stepY = 0.0;
    capy_pixel_steps_from_zoom_scale(sx, sy, width, (double)zoom, stepX, stepY);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CAPY][ARGS] cx=%.9f cy=%.9f stepX=%.11f stepY=%.11f it=%d w=%d h=%d",
                       cx, cy, stepX, stepY, maxIterations, width, height);
    }

    if constexpr (Settings::performanceLogging) {
        ensureEventsOnce();
        auto rc = cudaEventRecord(s_evStart, renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(start) before capy_render", rc);
    }

    capy_render(
        static_cast<uint16_t*>(d_iterations.get()),
        width, height, cx, cy, stepX, stepY,
        maxIterations, renderStream, state.evEcDone
    );

    auto rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("capy_render launch", rc);

    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStop, renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after capy_render", rc);
        rc = cudaEventSynchronize(s_evStop);
        if (rc != cudaSuccess) throw_with_log("capy_render sync", rc);
    }

    // 3) colorize into mapped PBO
    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStart, renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(start) before colorize", rc);
    }

    colorize_iterations_to_pbo(
        static_cast<const uint16_t*>(d_iterations.get()),
        static_cast<uchar4*>(map.ptr),
        width, height, maxIterations, renderStream
    );

    rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("colorize launch", rc);

    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStop, renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after colorize", rc);
        rc = cudaEventSynchronize(s_evStop);
        if (rc != cudaSuccess) throw_with_log("colorize sync", rc);
    }
}

// optional debug: iterations -> host mirror
bool downloadIterationsToHost(RendererState& state,
                              int width, int height,
                              std::vector<uint16_t>& host,
                              cudaStream_t stream) noexcept
{
    if (width <= 0 || height <= 0) return false;
    const size_t n = (size_t)width * (size_t)height;
    const size_t bytes = n * sizeof(uint16_t);
    host.resize(n);

    const void* dptr = state.d_iterations.get();
    if (!dptr) return false;

    auto rc = cudaMemcpyAsync(host.data(), dptr, bytes, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] memcpyAsync iter->host rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }
    if (stream) {
        rc = cudaStreamSynchronize(stream);
        if (rc != cudaSuccess) {
            LUCHS_LOG_HOST("[HM][ERR] streamSync after memcpy rc=%d", (int)rc);
            LuchsLogger::flushDeviceLogToHost(0);
            return false;
        }
    }
    return true;
}

// keep old API, delegate to HeatmapMetrics
bool buildHeatmapMetrics(RendererState& state,
                         int width, int height, int tilePx,
                         cudaStream_t stream) noexcept
{
    return HeatmapMetrics::buildGPU(state, width, height, tilePx, stream);
}

// ---- Convenience overload (float offsets) ------------------------------------
void renderCudaFrame(RendererState& state, const FrameContext& fctx, float& newOffsetX, float& newOffsetY) {
    float offx = fctx.offset.x;
    float offy = fctx.offset.y;
    bool  shouldZoom = false;

    renderCudaFrame(state.d_iterations,
                    fctx.width, fctx.height, fctx.zoom,
                    offx, offy, fctx.maxIterations,
                    newOffsetX, newOffsetY, shouldZoom,
                    state, state.renderStream);
}

// ---- Convenience overload (double offsets) — WURZEL-FIX ----------------------
void renderCudaFrame(RendererState& state, const FrameContext& fctx,
                     double& newOffsetX, double& newOffsetY)
{
    // Dieser Overload ist analog zum Low-Level-Pfad, führt aber die Offsets als double
    // bis in den capy_render-Launch (keine float-Quantisierung bei tiefen Zooms).

    if (!s_pboActive) {
        LUCHS_LOG_HOST("[PBO][ERR] render(double) called without registered PBOs");
        state.skipUploadThisFrame = true;
        return;
    }
    const int width  = fctx.width;
    const int height = fctx.height;
    if (width <= 0 || height <= 0) {
        LUCHS_LOG_HOST("[CUDA][ERR] invalid framebuffer dims %dx%d", width, height);
        state.skipUploadThisFrame = true;
        return;
    }

    (void)cudaGetLastError(); // clear sticky

    // 1) map current PBO slot
    const size_t needBytes = size_t(width) * size_t(height) * sizeof(uchar4);
    const int ix = (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) ? state.pboIndex : 0;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][MAP] try ring=%d need=%zu", ix, (size_t)needBytes);
    }

    MapGuard map(&s_pboResources[ix]);

    if (!map.ptr) {
        const auto rcMap = cudaGetLastError();
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

    // 2) capybara render (iterations) — *double* Offsets
    const double cx = newOffsetX;
    const double cy = newOffsetY;

    const double sx = (double)state.pixelScale.x;
    const double sy = (double)state.pixelScale.y;
    double stepX = 0.0, stepY = 0.0;
    capy_pixel_steps_from_zoom_scale(sx, sy, width, (double)fctx.zoom, stepX, stepY);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CAPY][ARGS][dbl] cx=%.12f cy=%.12f stepX=%.12e stepY=%.12e it=%d w=%d h=%d",
                       cx, cy, stepX, stepY, fctx.maxIterations, width, height);
    }

    if constexpr (Settings::performanceLogging) {
        ensureEventsOnce();
        auto rc = cudaEventRecord(s_evStart, state.renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(start) before capy_render[dbl]", rc);
    }

    capy_render(
        static_cast<uint16_t*>(state.d_iterations.get()),
        width, height, cx, cy, stepX, stepY,
        fctx.maxIterations, state.renderStream, state.evEcDone
    );

    auto rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("capy_render launch[dbl]", rc);

    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStop, state.renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after capy_render[dbl]", rc);
        rc = cudaEventSynchronize(s_evStop);
        if (rc != cudaSuccess) throw_with_log("capy_render sync[dbl]", rc);
    }

    // 3) colorize into mapped PBO
    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStart, state.renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(start) before colorize[dbl]", rc);
    }

    colorize_iterations_to_pbo(
        static_cast<const uint16_t*>(state.d_iterations.get()),
        static_cast<uchar4*>(map.ptr),
        width, height, fctx.maxIterations, state.renderStream
    );

    rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) throw_with_log("colorize launch[dbl]", rc);

    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStop, state.renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(stop) after colorize[dbl]", rc);
        rc = cudaEventSynchronize(s_evStop);
        if (rc != cudaSuccess) throw_with_log("colorize sync[dbl]", rc);
    }

    // Hinweis: newOffsetX/newOffsetY werden hier absichtlich nicht verändert.
}

void logCudaContext(const char* tag) noexcept { logCudaDeviceContext(tag); }

} // namespace CudaInterop
