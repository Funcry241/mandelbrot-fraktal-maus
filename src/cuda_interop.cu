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
        cudaError_t rc = cudaEventCreate(&s_evStart);
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

    // ---- Heatmap metric buffers on device --------------------------------
    static float*  s_d_entropy  = nullptr;
    static float*  s_d_contrast = nullptr;
    static size_t  s_metricsN   = 0;

    static void ensureMetricsCapacity(size_t N) {
        if (N <= s_metricsN) return;
        if (s_d_entropy)  { cudaFree(s_d_entropy);  s_d_entropy  = nullptr; }
        if (s_d_contrast) { cudaFree(s_d_contrast); s_d_contrast = nullptr; }
        if (N) {
            cudaMalloc(&s_d_entropy,  N * sizeof(float));
            cudaMalloc(&s_d_contrast, N * sizeof(float));
        }
        s_metricsN = N;
    }

    // ---- RAII-Guard: map/unmap -------------------------------------------
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

    // ---- helpers ----------------------------------------------------------
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

    // --------- GPU kernel: per-tile entropy & contrast ---------------------
    __global__ void kernel_tile_metrics(const uint16_t* __restrict__ it,
                                        int w, int h, int tilePx, int tilesX,
                                        float* __restrict__ entropy,
                                        float* __restrict__ contrast)
    {
        const int tx = blockIdx.x, ty = blockIdx.y;
        if (tx >= tilesX) return;

        const int tilesY = gridDim.y;
        const int x0 = tx * tilePx;
        const int y0 = ty * tilePx;
        int x1 = x0 + tilePx; if (x1 > w) x1 = w;
        int y1 = y0 + tilePx; if (y1 > h) y1 = h;

        __shared__ unsigned int hist[256];
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();

        for (int y = y0 + threadIdx.y; y < y1; y += blockDim.y) {
            for (int x = x0 + threadIdx.x; x < x1; x += blockDim.x) {
                const uint16_t v = it[y * w + x];
                // 12->8 bit bucketing, saturiert
                unsigned b = (unsigned)(v >> 4); if (b > 255u) b = 255u;
                atomicAdd(&hist[b], 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            unsigned int cnt = 0u;
            for (int i=0;i<256;i++) cnt += hist[i];
            const float inv = cnt ? 1.0f / cnt : 0.0f;

            float H = 0.f, mu = 0.f, var = 0.f;
            for (int i=0;i<256;i++) {
                const float p = hist[i] * inv;
                if (p > 0.f) H -= p * log2f(p);
                mu += i * p;
            }
            for (int i=0;i<256;i++) {
                const float p = hist[i] * inv;
                const float d = i - mu;
                var += d * d * p;
            }

            const int idx = ty * tilesX + tx;
            entropy[idx]  = H;
            // leichte Normierung, damit ~[0..1] grob passt
            contrast[idx] = sqrtf(var) / 128.f;
        }
    }

    // Compute metrics on GPU and copy to host vectors in RendererState.
    static void compute_metrics_and_copy_to_host(const uint16_t* d_it,
                                                 int width, int height,
                                                 int tilePx,
                                                 RendererState& state,
                                                 cudaStream_t stream)
    {
        const int px = (tilePx > 0) ? tilePx : 1;
        const int tilesX = (width  + px - 1) / px;
        const int tilesY = (height + px - 1) / px;
        const size_t N   = (size_t)tilesX * (size_t)tilesY;
        ensureMetricsCapacity(N);

        dim3 grid(tilesX, tilesY);
        dim3 block(16,16);
        kernel_tile_metrics<<<grid, block, 0, stream>>>(
            d_it, width, height, px, tilesX,
            s_d_entropy, s_d_contrast
        );
        cudaError_t rcM = cudaPeekAtLastError();
        if (rcM != cudaSuccess) throw_with_log("metrics launch", rcM);

        // sync + copy to host
        rcM = cudaStreamSynchronize(stream);
        if (rcM != cudaSuccess) throw_with_log("metrics sync", rcM);

        state.h_entropy.resize(N);
        state.h_contrast.resize(N);
        cudaMemcpy(state.h_entropy.data(),  s_d_entropy,  N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(state.h_contrast.data(), s_d_contrast, N*sizeof(float), cudaMemcpyDeviceToHost);

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[HM][GPU] tiles=%dx%d N=%zu tilePx=%d", tilesX, tilesY, N, px);
        }
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
    s_pboResources.clear();
    s_pboActive = false;
    // free metric buffers too
    if (s_d_entropy)  { cudaFree(s_d_entropy);  s_d_entropy  = nullptr; }
    if (s_d_contrast) { cudaFree(s_d_contrast); s_d_contrast = nullptr; }
    s_metricsN = 0;
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

// ----------------------------------------------------------------------
// Hauptpfad: render -> colorize to PBO
// ----------------------------------------------------------------------
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

    ensureEventsOnce();
    (void)cudaGetLastError();

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

    // ---- Capybara render (iterations) ----
    const double cx   = (double)offsetX;
    const double cy   = (double)offsetY;

    const double sx = (double)state.pixelScale.x;
    const double sy = (double)state.pixelScale.y;

    // Schrittweite mit 1/zoom skalieren
    double baseStep = std::max(std::abs(sx), std::abs(sy));
    if (!(baseStep > 0.0)) {
        constexpr double kBaseSpan = 8.0 / 3.0; // Field-of-view Basisspanne
        baseStep = kBaseSpan / std::max(1, width);
    }
    const double invZ  = 1.0 / std::max(1.0, (double)zoom);
    const double step  = baseStep * invZ;

    const double stepX = std::copysign(step, ( sx == 0.0 ?  1.0 : sx));
    const double stepY = std::copysign(step, ( sy == 0.0 ? -1.0 : sy));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CAPY][ARGS] cx=%.9f cy=%.9f stepX=%.11f stepY=%.11f it=%d w=%d h=%d",
                       cx, cy, stepX, stepY, maxIterations, width, height);
        LUCHS_LOG_HOST("[CAPY][STEP] zoom=%.6f invZ=%.6f baseStep=%.10f", (double)zoom, invZ, baseStep);
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

    // ---- Colorize into mapped PBO ----
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

// ABI-kompatibler Wrapper: hier hängen wir die GPU-Heatmap dran
void renderCudaFrame(RendererState& state, const FrameContext& fctx, float& newOffsetX, float& newOffsetY) {
    float offx = fctx.offset.x;
    float offy = fctx.offset.y;
    bool  shouldZoom = false;

    // 1) Capybara Render + Colorize
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

    // 2) GPU-Heatmap berechnen & hostseitig bereitstellen (Overlay erwartet die Vektoren)
    const int tilePx = (Settings::Kolibri::gridScreenConstant)
                     ? std::max(1, Settings::Kolibri::desiredTilePx)
                     : std::max(1, fctx.tileSize);

    compute_metrics_and_copy_to_host(
        static_cast<const uint16_t*>(state.d_iterations.get()),
        fctx.width, fctx.height, tilePx,
        state,
        state.renderStream
    );
}

void logCudaContext(const char* tag) noexcept { logCudaDeviceContext(tag); }

// Optionales Debug-Tooling: Iterationsbuffer -> Host spiegeln
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

    cudaError_t rc = cudaMemcpyAsync(host.data(), dptr, bytes, cudaMemcpyDeviceToHost, stream);
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

} // namespace CudaInterop
