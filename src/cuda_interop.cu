///// Otter: OpenGL PBO interop; map/unmap + pointer retrieval logged deterministically.
///// Schneefuchs: Precheck cuda runtime; numeric rc codes only; no getErrorString.
///// Maus: Immediate device-log flush on CUDA errors; one line per event.
///// Datei: src/cuda_interop.cu

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"   // LuchsLogger::flushDeviceLogToHost
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#if !defined(__CUDA_ARCH__)
  #include <chrono>
  #include <iomanip>
#endif

// ---- Minimal GL-Forward-Decls (keine schweren GL-Header nötig) --------------
struct __GLsync; using GLsync = __GLsync*;

// Nur definieren, wenn KEIN GL-/GLEW-Header aktiv ist (verhindert Redefinitionen)
#if !defined(__gl_h_) && !defined(GLEW_H) && !defined(__glew_h__)
  using GLuint     = unsigned int;
  using GLenum     = unsigned int;
  using GLbitfield = unsigned int;
  using GLuint64   = unsigned long long;
  extern "C" {
      GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);
      void   glDeleteSync(GLsync sync);
  }
#endif

// ---- Kernel (extern C) ------------------------------------------------------
// Stream-Parameter in der Deklaration (Render-Stream kommt von außen)
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, uint16_t* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int tile,
    cudaStream_t stream
);

namespace CudaInterop {

// ---- TU-lokaler Zustand -----------------------------------------------------
static bear_CudaPBOResource*                     s_pboActive = nullptr;
static std::unordered_map<GLuint, bear_CudaPBOResource*> s_pboMap;

static bool           s_pauseZoom = false;
static bool           s_deviceOk  = false;

static cudaEvent_t    s_evStart = nullptr, s_evStop = nullptr; static bool s_evInit = false;

// ---- Helpers ----------------------------------------------------------------
static inline void ensureDeviceOnce() {
    if (!s_deviceOk) { CUDA_CHECK(cudaSetDevice(0)); s_deviceOk = true; }
}

static inline void ensureEventsOnce() {
    if (s_evInit) return;
    CUDA_CHECK(cudaEventCreate(&s_evStart));
    CUDA_CHECK(cudaEventCreate(&s_evStop));
    s_evInit = (s_evStart && s_evStop);
    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[CUDA][ZK] events %s", s_evInit ? "created" : "FAILED");
}

static inline void destroyEventsIfAny() {
    if (!s_evInit) return;
    cudaEventDestroy(s_evStart); s_evStart=nullptr;
    cudaEventDestroy(s_evStop);  s_evStop =nullptr;
    s_evInit=false;
}

static inline void enforceWriteDiscard(bear_CudaPBOResource* res) {
    if (!res) return;
    if (auto* gr = res->get()) {
        (void)cudaGraphicsResourceSetMapFlags(gr, cudaGraphicsMapFlagsWriteDiscard);
    }
}

struct MapGuard {
    bear_CudaPBOResource* r=nullptr;
    void* ptr=nullptr; size_t bytes=0;
    explicit MapGuard(bear_CudaPBOResource* rr):r(rr){ if(r){ ptr=r->mapAndLog(bytes);} }
    ~MapGuard(){ if(r) r->unmap(); }
    MapGuard(const MapGuard&) = delete; MapGuard& operator=(const MapGuard&) = delete;
};

// ---- PBO-Verwaltung ---------------------------------------------------------
void registerAllPBOs(const GLuint* ids, int count) {
    ensureDeviceOnce();

    destroyEventsIfAny();

    for (auto &kv : s_pboMap) delete kv.second; s_pboMap.clear(); s_pboActive=nullptr;
    if (!ids || count<=0) return;

    for (int i=0;i<count;++i) {
        if (!ids[i]) continue;
        auto* res = new bear_CudaPBOResource(ids[i]);
        if (res && res->get()) {
            enforceWriteDiscard(res);
            s_pboMap[ids[i]] = res;
        } else {
            delete res;
        }
    }
    for (int i=0;i<count && !s_pboActive;++i){ auto it=s_pboMap.find(ids[i]); if(it!=s_pboMap.end()) s_pboActive=it->second; }
}

void unregisterAllPBOs() {
    destroyEventsIfAny();

    for (auto &kv : s_pboMap) delete kv.second; s_pboMap.clear(); s_pboActive=nullptr;
}

void registerPBO(const Hermelin::GLBuffer& pbo) {
    ensureDeviceOnce();
    const GLuint id = pbo.id();
    auto it = s_pboMap.find(id);
    if (it == s_pboMap.end()) {
        auto* res = new bear_CudaPBOResource(id);
        if (res && res->get()) {
            enforceWriteDiscard(res);
            s_pboMap[id]=res;
            if constexpr (Settings::debugLogging)
                LUCHS_LOG_HOST("[CUDA-Interop] auto-registered PBO id=%u", id);
        } else {
            delete res;
            LUCHS_LOG_HOST("[FATAL] failed to create CudaPBOResource id=%u", id);
            return;
        }
        it = s_pboMap.find(id);
    }
    s_pboActive = it->second;
}

void unregisterPBO() {
    destroyEventsIfAny();

    if (s_pboActive) {
        for (auto it = s_pboMap.begin(); it != s_pboMap.end(); ++it) {
            if (it->second == s_pboActive) { delete it->second; s_pboMap.erase(it); break; }
        }
        s_pboActive = nullptr;
    }
}

// ---- Hauptpfad --------------------------------------------------------------
void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
    int width, int height,
    float zoom, float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float2& newOffset, bool& shouldZoom,
    int tileSize, RendererState& state,
    cudaStream_t renderStream,
    cudaStream_t copyStream
){
#if !defined(__CUDA_ARCH__)
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs=0.0, mbMs=0.0;
#endif
    if (!s_pboActive) throw std::runtime_error("[FATAL] CUDA PBO not registered!");
    if (width<=0 || height<=0)  throw std::runtime_error("invalid framebuffer dims");
    if (tileSize<=0) { int was=tileSize; tileSize = Settings::BASE_TILE_SIZE>0 ? Settings::BASE_TILE_SIZE : 16; LUCHS_LOG_HOST("[WARN] tileSize<=0 (%d) -> using %d", was, tileSize); }

    const size_t totalPx = size_t(width)*size_t(height);
    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    const size_t itBytes = totalPx * sizeof(uint16_t);
    const size_t enBytes = size_t(numTiles) * sizeof(float);
    const size_t ctBytes = size_t(numTiles) * sizeof(float);

    if (d_iterations.size()<itBytes || d_entropy.size()<enBytes || d_contrast.size()<ctBytes)
        throw std::runtime_error("CudaInterop::renderCudaFrame: device buffers undersized");

    // [LOG-1] Pre-Map Fence-Check & deterministisches Skippen des Uploads
    {
        GLsync f = state.pboFence[state.pboIndex];
        if (f) {
            // non-blocking Poll; wir wollen nicht bis zum Timeout warten
            GLenum r = glClientWaitSync(f, 0, 0);
            if (r == 0x9111 /*GL_TIMEOUT_EXPIRED*/ || r == 0 /*GL_WAIT_FAILED (0) als konservativer Check*/ ) {
                state.skipUploadThisFrame = true;
                if (state.pboIndex >= 0 && state.pboIndex < RendererState::kPboRingSize) {
                    state.ringSkip++; // Statistik
                }
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ZK][UP] pre-map fence busy → skip upload this frame (ring=%d)", state.pboIndex);
                }
                return; // Kein Map, kein Render/Upload in diesem Frame
            }
            // Fence ist signaled → wegräumen
            glDeleteSync(f);
            state.pboFence[state.pboIndex] = 0;
        }
    }

#if !defined(__CUDA_ARCH__)
    const auto tMap0 = std::chrono::high_resolution_clock::now();
#endif
    MapGuard map(s_pboActive);
    if (!map.ptr) throw std::runtime_error("pboResource->map() returned null");

#if !defined(__CUDA_ARCH__)
    const auto tMap1 = std::chrono::high_resolution_clock::now();
    mapMs = std::chrono::duration<double, std::milli>(tMap1 - tMap0).count();
#endif
    const size_t needBytes = size_t(width)*size_t(height)*sizeof(uchar4);
    if (map.bytes < needBytes) throw std::runtime_error("PBO byte size mismatch");

    // Ring-Statistik: erfolgreiche Nutzung dieses Slots
    if (state.pboIndex >= 0 && state.pboIndex < RendererState::kPboRingSize) {
        state.ringUse[state.pboIndex]++;
    }

    ensureEventsOnce();
    (void)cudaGetLastError();

    // Timing-Event auf DEM Render-Stream (nicht Stream 0)
    CUDA_CHECK(cudaEventRecord(s_evStart, renderStream));

    // Kernel-Launch auf dem übergebenen Stream
    launch_mandelbrotHybrid(static_cast<uchar4*>(map.ptr),
                            static_cast<uint16_t*>(d_iterations.get()),
                            width, height, zoom, offset, maxIterations, tileSize,
                            renderStream);
    cudaError_t mbErrLaunch = cudaGetLastError();

    // Stop-Event & Sync ebenfalls auf renderStream
    CUDA_CHECK(cudaEventRecord(s_evStop, renderStream));
    cudaError_t mbErrSync = cudaEventSynchronize(s_evStop);

#if !defined(__CUDA_ARCH__)
    if (mbErrSync==cudaSuccess) {
        float ms=0.0f; cudaEventElapsedTime(&ms, s_evStart, s_evStop); mbMs = ms;
    }
#endif
    if (mbErrLaunch != cudaSuccess || mbErrSync != cudaSuccess) {
        LUCHS_LOG_HOST("[CUDA][ERR] mandelbrot rc_launch=%d rc_sync=%d", (int)mbErrLaunch, (int)mbErrSync);
        // [LOG-3] Sofort Device-Log flushen (wenn initialisiert; hier best-effort)
        LuchsLogger::flushDeviceLogToHost(0);
        throw std::runtime_error("CUDA failure: mandelbrot kernel");
    }

    // ---------------- E/C-Streaming (Minimal: 2 Streams) ----------------
    // E/C läuft auf dem renderStream hinter dem Mandelbrot-Render und setzt ein Done-Event.
    ::computeCudaEntropyContrast(
        static_cast<const uint16_t*>(d_iterations.get()),
        static_cast<float*>(d_entropy.get()),
        static_cast<float*>(d_contrast.get()),
        width, height, tileSize, maxIterations,
        renderStream,                // Launch-Stream
        state.evEcDone              // record done-event
    );

    // Host-Transfers (copyStream wartet auf E/C-done Event)
    if (h_entropy.capacity()  < size_t(numTiles)) h_entropy.reserve(size_t(numTiles));
    if (h_contrast.capacity() < size_t(numTiles)) h_contrast.reserve(size_t(numTiles));
    h_entropy.resize(size_t(numTiles));
    h_contrast.resize(size_t(numTiles));

    CUDA_CHECK(cudaStreamWaitEvent(copyStream, state.evEcDone, 0));

    CUDA_CHECK(cudaMemcpyAsync(h_entropy.data(),  d_entropy.get(),  enBytes, cudaMemcpyDeviceToHost, copyStream));
    CUDA_CHECK(cudaMemcpyAsync(h_contrast.data(), d_contrast.get(), ctBytes, cudaMemcpyDeviceToHost, copyStream));

    if (state.evCopyDone) {
        CUDA_CHECK(cudaEventRecord(state.evCopyDone, copyStream));
    }

    // KEIN spätes Synchronisieren mehr: call-site wartet auf evCopyDone
    shouldZoom = false; newOffset = offset;

#if !defined(__CUDA_ARCH__)
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // [LOG-2] Timings in State übertragen (hier: PBO-Map & Mandelbrot total)
    state.lastTimings.valid            = true;
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = mbMs;
    state.lastTimings.mandelbrotLaunch = 0.0;
    state.lastTimings.mandelbrotSync   = 0.0;
    state.lastTimings.entropy          = 0.0; // exakte E/C über Events separat messen (andere TU)
    state.lastTimings.contrast         = 0.0;
    state.lastTimings.deviceLogFlush   = 0.0;

    if constexpr (Settings::performanceLogging)
        LUCHS_LOG_HOST("[PERF][ZK] map=%.2f mandelbrot=%.2f entropy=%.2f contrast=%.2f total=%.2f",
                       mapMs, mbMs, state.lastTimings.entropy, state.lastTimings.contrast, totalMs);
#endif
}

// ---- Sonstiges API ----------------------------------------------------------
void setPauseZoom(bool pause){ s_pauseZoom = pause; }
bool getPauseZoom(){ return s_pauseZoom; }

bool precheckCudaRuntime() {
    int deviceCount = 0;
    cudaError_t e1 = cudaFree(0);
    cudaError_t e2 = cudaGetDeviceCount(&deviceCount);
    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[CUDA] precheck err1=%d err2=%d count=%d", (int)e1, (int)e2, deviceCount);
    return e1==cudaSuccess && e2==cudaSuccess && deviceCount>0;
}

static inline int getAttrSafe(cudaDeviceAttr a, int dev){ int v=0; (void)cudaDeviceGetAttribute(&v,a,dev); return v; }
void logCudaDeviceContext(const char* tag) {
    if constexpr (!(Settings::debugLogging || Settings::performanceLogging)) { (void)tag; return; }
    int dev=-1; cudaError_t e0=cudaGetDevice(&dev);
    int rt=0, drv=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    char name[256]={0};
    if (dev>=0){ cudaDeviceProp p{}; if (cudaGetDeviceProperties(&p,dev)==cudaSuccess) std::strncpy(name,p.name,sizeof(name)-1); }
    if (e0==cudaSuccess && dev>=0) {
        const int ccM=getAttrSafe(cudaDevAttrComputeCapabilityMajor,dev);
        const int ccN=getAttrSafe(cudaDevAttrComputeCapabilityMinor,dev);
        const int sms=getAttrSafe(cudaDevAttrMultiProcessorCount,dev);
        const int warp=getAttrSafe(cudaDevAttrWarpSize,dev);
        size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
        LUCHS_LOG_HOST("[CUDA] ctx tag=%s rt=%d drv=%d dev=%d name=\"%s\" cc=%d.%d sms=%d warp=%d memMB free=%zu total=%zu",
            (tag?tag:"(null)"), rt, drv, dev, name, ccM, ccN, sms, warp, (mf>>20), (mt>>20));
    } else {
        LUCHS_LOG_HOST("[CUDA] ctx tag=%s deviceQuery failed e0=%d dev=%d", (tag?tag:"(null)"), (int)e0, dev);
    }
}

} // namespace CudaInterop
