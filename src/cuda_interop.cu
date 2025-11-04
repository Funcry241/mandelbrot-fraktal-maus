///// Otter: OpenGL PBO interop – fence-aware map; single Capybara core; deterministic logs
///// Schneefuchs: Ring-Slotwahl mit GLsync; numeric CUDA rc; Events nur bei PerfLog; keine Dup-Pfade
///// Maus: Skip statt Blockieren bei Sättigung; klare [ZK]-Logs; Heatmap bleibt delegiert
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
    ~MapGuard() noexcept {
        if (res) res->unmapAndLog();
    }
    MapGuard(const MapGuard&) = delete;
    MapGuard& operator=(const MapGuard&) = delete;
    MapGuard(MapGuard&&) = delete;
    MapGuard& operator=(MapGuard&&) = delete;
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
        if ((int)s_pboResources.size() != RendererState::kPboRingSize) {
            LUCHS_LOG_HOST("[PBO][WARN] resources=%d != ringSize=%d",
                           (int)s_pboResources.size(), RendererState::kPboRingSize);
        }
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

// ------------------------------ fence-aware PBO slotwahl ------------------------------
static int choose_free_pbo_index(RendererState& state) {
    // Start beim gewünschten Index; nacheinander Slots prüfen.
    const int N = RendererState::kPboRingSize;
    int start = (state.pboIndex >= 0 && state.pboIndex < N) ? state.pboIndex : 0;

    for (int k = 0; k < N; ++k) {
        const int ix = (start + k) % N;
        GLsync f = state.pboFence[ix];
        if (!f) {
            // Kein Fence -> Slot ist frei
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZK][PBO] pick free slot=%d (no fence)", ix);
            }
            return ix;
        }
        // Non-blocking prüfen
        const GLenum st = glClientWaitSync(f, 0, 0);
        if (st == GL_ALREADY_SIGNALED || st == GL_CONDITION_SATISFIED) {
            // Fertig – Fence aufräumen
            glDeleteSync(f);
            state.pboFence[ix] = 0;
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZK][PBO] pick slot=%d (fence signaled)", ix);
            }
            return ix;
        }
        // TIMEOUT_EXPIRED / WAIT_FAILED -> Slot busy, weiter prüfen
    }
    // Alle Slots busy → lieber Frame überspringen statt blockieren
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZK][PBO][SAT] ring saturated – skip upload this frame");
    }
    return -1;
}

// ------------------------------ single render core ------------------------------
/*
   Nacktmull: one authoritative core that maps the current (fence-free) PBO,
   runs capy_render, then colorizes into the mapped memory.
*/
static void render_to_pbo_core(RendererState& state,
                               int width, int height,
                               double cx, double cy,
                               double zoom,
                               int maxIterations,
                               cudaStream_t renderStream)
{
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

    // 0) Zaunkönig: fence-aware Slotwahl
    const int freeIx = choose_free_pbo_index(state);
    if (freeIx < 0) {
        state.skipUploadThisFrame = true;
        return; // Keine Compute/Colorize, um Stalls zu vermeiden
    }
    // Auf den tatsächlich nutzbaren Slot umstellen
    state.pboIndex = freeIx;

    // 1) map current PBO slot
    const size_t needBytes = size_t(width) * size_t(height) * sizeof(uchar4);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][MAP] try ring=%d need=%zu", state.pboIndex, (size_t)needBytes);
    }

    MapGuard map(&s_pboResources[state.pboIndex]);

    if (!map.ptr) {
        const auto rcMap = cudaGetLastError();
        LUCHS_LOG_HOST("[PBO][MAP][ERR] null ptr ring=%d need=%zu rc=%d", state.pboIndex, (size_t)needBytes, (int)rcMap);
        LuchsLogger::flushDeviceLogToHost(0);
        state.skipUploadThisFrame = true;
        return;
    }
    if (map.bytes < needBytes) {
        LUCHS_LOG_HOST("[PBO][MAP][ERR] size mismatch ring=%d got=%zu need=%zu",
                       state.pboIndex, (size_t)map.bytes, (size_t)needBytes);
        LuchsLogger::flushDeviceLogToHost(0);
        state.skipUploadThisFrame = true;
        return;
    }

    if (state.pboIndex >= 0 && state.pboIndex < (int)s_pboResources.size()) {
        state.ringUse[state.pboIndex]++;
    }

    // 2) capybara render (iterations)
    const double sx = (double)state.pixelScale.x;
    const double sy = (double)state.pixelScale.y;
    double stepX = 0.0, stepY = 0.0;
    capy_pixel_steps_from_zoom_scale(sx, sy, width, zoom, stepX, stepY);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CAPY][ARGS] cx=%.12f cy=%.12f stepX=%.12e stepY=%.12e it=%d w=%d h=%d",
                       cx, cy, stepX, stepY, maxIterations, width, height);
    }

    if constexpr (Settings::performanceLogging) {
        ensureEventsOnce();
        auto rc = cudaEventRecord(s_evStart, renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(start) before capy_render", rc);
    }

    capy_render(
        static_cast<uint16_t*>(state.d_iterations.get()),
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
        float ms = 0.0f;
        (void)cudaEventElapsedTime(&ms, s_evStart, s_evStop);
        LUCHS_LOG_HOST("[CAPY][time] capy_render=%.3f ms (w=%d h=%d it=%d)", (double)ms, width, height, maxIterations);
    }

    // 3) colorize into mapped PBO
    if constexpr (Settings::performanceLogging) {
        rc = cudaEventRecord(s_evStart, renderStream);
        if (rc != cudaSuccess) throw_with_log("eventRecord(start) before colorize", rc);
    }

    colorize_iterations_to_pbo(
        static_cast<const uint16_t*>(state.d_iterations.get()),
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
        float ms = 0.0f;
        (void)cudaEventElapsedTime(&ms, s_evStart, s_evStop);
        LUCHS_LOG_HOST("[CAPY][time] colorize=%.3f ms (w=%d h=%d)", (double)ms, width, height);
    }
}

// ------------------------------ public API (forwards) ------------------------------
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
    (void)d_iterations; // authoritative buffer lives in RendererState
    shouldZoom = false;
    newOffsetX = offsetX;
    newOffsetY = offsetY;

    render_to_pbo_core(state, width, height,
                       (double)offsetX, (double)offsetY,
                       (double)zoom, maxIterations, renderStream);
}

// Convenience overload (double offsets) – hohe Präzision, KEINE Zoom-Logik
void renderCudaFrame(RendererState& state, const FrameContext& fctx,
                     double& newOffsetX, double& newOffsetY)
{
    (void)s_pauseZoom; // Flag wird nur zentral in FramePipeline beachtet

    const int    width  = fctx.width;
    const int    height = fctx.height;
    const double cx     = (double)state.center.x; // authoritative from RendererState
    const double cy     = (double)state.center.y;
    const double zoom   = (double)state.zoom;

    render_to_pbo_core(state, width, height, cx, cy, zoom, fctx.maxIterations, state.renderStream);

    // Für Aufrufer/Telemetry zurückspiegeln
    newOffsetX = cx;
    newOffsetY = cy;
}

// keep old API, delegate to HeatmapMetrics
bool buildHeatmapMetrics(RendererState& state,
                         int width, int height, int tilePx,
                         cudaStream_t stream) noexcept
{
    return HeatmapMetrics::buildGPU(state, width, height, tilePx, stream);
}

void logCudaContext(const char* tag) noexcept { logCudaDeviceContext(tag); }

} // namespace CudaInterop
