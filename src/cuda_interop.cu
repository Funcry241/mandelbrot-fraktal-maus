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
#include "capybara_mapping.cuh"  // capy_pixel_steps_from_zoom_scale(...)

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

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

// ======================================================================
// GPU-Heatmap-Metriken (Kontrast/Entropie) – Device-Teil
// ======================================================================

__global__ void kernel_tile_metrics(const uint16_t* __restrict__ it,
                                    int w, int h,
                                    int tilePx, int tilesX,
                                    float* __restrict__ entropy,
                                    float* __restrict__ contrast)
{
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int tilesY = gridDim.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int x0 = tx * tilePx;
    const int y0 = ty * tilePx;
    int x1 = x0 + tilePx; if (x1 > w) x1 = w;
    int y1 = y0 + tilePx; if (y1 > h) y1 = h;

    const int tileW = max(0, x1 - x0);
    const int tileH = max(0, y1 - y0);
    const int nPix  = tileW * tileH;
    const int outIx = ty * tilesX + tx;

    if (nPix <= 0) {
        if (entropy)  entropy[outIx]  = 0.0f;
        if (contrast) contrast[outIx] = 0.0f;
        return;
    }

    // Std-Abw. als Kontrast
    double sum = 0.0, sum2 = 0.0;
    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            const double v = (double)row[x];
            sum  += v; sum2 += v * v;
        }
    }
    const double mean = sum / (double)nPix;
    double var = sum2 / (double)nPix - mean * mean;
    if (var < 0.0) var = 0.0;
    if (contrast) contrast[outIx] = (float)sqrt(var);

    // Entropie über 32 Buckets (hash-basiert)
    constexpr int B = 32;
    int hist[B];
    #pragma unroll
    for (int i = 0; i < B; ++i) hist[i] = 0;

    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            const uint16_t v = row[x];
            const int b = ((int)v ^ ((int)v >> 5)) & (B - 1);
            hist[b] += 1;
        }
    }

    float H = 0.0f;
    const float invN = 1.0f / (float)nPix;
    constexpr float invLn2 = 1.0f / 0.6931471805599453f;
    for (int i = 0; i < B; ++i) {
        const float p = (float)hist[i] * invN;
        if (p > 0.0f) H -= p * (logf(p) * invLn2);
    }
    if (entropy) entropy[outIx] = H;
}

// Device-Cache für Metriken
static float* s_dEntropy  = nullptr;
static float* s_dContrast = nullptr;
static size_t s_tilesCap  = 0;

static bool ensureTileMetricBuffers(size_t tilesNeeded) {
    if (tilesNeeded <= s_tilesCap && s_dEntropy && s_dContrast) return true;

    if (s_dEntropy)  cudaFree(s_dEntropy);
    if (s_dContrast) cudaFree(s_dContrast);
    s_dEntropy = s_dContrast = nullptr;
    s_tilesCap = 0;

    cudaError_t rc = cudaMalloc((void**)&s_dEntropy,  tilesNeeded * sizeof(float));
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] cudaMalloc dEntropy tiles=%zu rc=%d", tilesNeeded, (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }
    rc = cudaMalloc((void**)&s_dContrast, tilesNeeded * sizeof(float));
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] cudaMalloc dContrast tiles=%zu rc=%d", tilesNeeded, (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        cudaFree(s_dEntropy); s_dEntropy = nullptr;
        return false;
    }
    s_tilesCap = tilesNeeded;
    return true;
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
    (void)newOffsetX; (void)newOffsetY; (void)shouldZoom;

    if (!s_pboActive) {
        LUCHS_LOG_HOST("[PBO][ERR] render called without registered PBOs]");
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
        cudaError_t rcMap = cudaGetLastError();
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

    // ---- 2) Capybara render (iterations) --------------------------------
    const double cx   = (double)offsetX;
    const double cy   = (double)offsetY;

    const double sx = (double)state.pixelScale.x;
    const double sy = (double)state.pixelScale.y;
    double stepX = 0.0, stepY = 0.0;
    capy_pixel_steps_from_zoom_scale(sx, sy, width, (double)zoom, stepX, stepY);

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
        state.evEcDone
    );

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
}

// GPU-Heatmap in Host-Vektoren spiegeln
bool buildHeatmapMetrics(RendererState& state,
                         int width, int height, int tilePx,
                         cudaStream_t stream) noexcept
{
    if (width <= 0 || height <= 0 || tilePx <= 0) return false;

    const int px = std::max(1, tilePx);
    const int tilesX = (width  + px - 1) / px;
    const int tilesY = (height + px - 1) / px;
    const size_t tiles = (size_t)tilesX * (size_t)tilesY;

    if (!ensureTileMetricBuffers(tiles)) return false;

    dim3 grid((unsigned)tilesX, (unsigned)tilesY, 1);
    dim3 block(1, 1, 1);

    kernel_tile_metrics<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(state.d_iterations.get()),
        width, height,
        px, tilesX,
        s_dEntropy, s_dContrast
    );
    cudaError_t rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] kernel launch rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    state.h_entropy.resize(tiles);
    state.h_contrast.resize(tiles);

    rc = cudaMemcpyAsync(state.h_entropy.data(),  s_dEntropy,  tiles * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(state.h_contrast.data(), s_dContrast, tiles * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] memcpyAsync metrics->host rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    rc = cudaStreamSynchronize(stream);
    if (rc != cudaSuccess) {
        LUCHS_LOG_HOST("[HM][ERR] streamSync metrics rc=%d", (int)rc);
        LuchsLogger::flushDeviceLogToHost(0);
        return false;
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM][GPU] ok tiles=%dx%d N=%zu tilePx=%d", tilesX, tilesY, tiles, px);
    }
    return true;
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

void logCudaContext(const char* tag) noexcept { logCudaDeviceContext(tag); }

} // namespace CudaInterop
