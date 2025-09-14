///// Otter: Kürzer & robust: Auto-PBO-Register, MapGuard, Events+Stream einmalig, ASCII-Perf-Logs.
/// /// Schneefuchs: Ein Pfad, deterministische State-Übergaben, konsistente Fehlerpfade, keine Redundanz.
/// /// Maus: Keine versteckten Pfade; Host-Pinning für schnelle memcpy; Shader/GL bleiben unberührt.
/// /// Datei: src/cuda_interop.cu

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#if !defined(__CUDA_ARCH__)
  #include <chrono>
#endif

// ---- Kernel (extern C) ------------------------------------------------------
extern "C" void launch_mandelbrotHybrid(
    uchar4* out, uint16_t* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int tile
);

namespace CudaInterop {

// ---- TU-lokaler Zustand -----------------------------------------------------
static bear_CudaPBOResource*                     s_pboActive = nullptr;
static std::unordered_map<GLuint, bear_CudaPBOResource*> s_pboMap;

static bool           s_pauseZoom = false;
static bool           s_deviceOk  = false;

static void*          s_hostRegEntropyPtr   = nullptr;  static size_t s_hostRegEntropyBytes   = 0;
static void*          s_hostRegContrastPtr  = nullptr;  static size_t s_hostRegContrastBytes  = 0;

static cudaEvent_t    s_evStart = nullptr, s_evStop = nullptr; static bool s_evInit = false;
static cudaStream_t   s_copyStrm = nullptr;

// ---- Helpers ----------------------------------------------------------------
static inline void ensureDeviceOnce() {
    if (!s_deviceOk) { CUDA_CHECK(cudaSetDevice(0)); s_deviceOk = true; }
}
static inline void ensureEventsOnce() {
    if (s_evInit) return;
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        cudaEventCreate(&s_evStart);
        cudaEventCreate(&s_evStop);
        s_evInit = (s_evStart && s_evStop);
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[CUDA] timing events %s", s_evInit ? "created" : "FAILED");
    }
}
static inline void ensureCopyStreamOnce() {
    if (!s_copyStrm) CUDA_CHECK(cudaStreamCreateWithFlags(&s_copyStrm, cudaStreamNonBlocking));
}
static inline void destroyEventsIfAny() {
    if (!s_evInit) return;
    cudaEventDestroy(s_evStart); s_evStart=nullptr;
    cudaEventDestroy(s_evStop);  s_evStop =nullptr;
    s_evInit=false;
}
static inline void ensureHostPinned(std::vector<float>& vec, void*& regPtr, size_t& regBytes) {
    const size_t cap = vec.capacity();
    void* ptr = cap ? (void*)vec.data() : nullptr;
    const size_t bytes = cap * sizeof(float);
    if (ptr == regPtr && bytes == regBytes) return;
    if (regPtr) CUDA_CHECK(cudaHostUnregister(regPtr));
    if (ptr)    CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
    regPtr = ptr; regBytes = bytes;
    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIN] host-register ptr=%p bytes=%zu", regPtr, regBytes);
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
    for (auto &kv : s_pboMap) delete kv.second; s_pboMap.clear(); s_pboActive=nullptr;
    if (!ids || count<=0) return;
    for (int i=0;i<count;++i) {
        if (!ids[i]) continue;
        auto* res = new bear_CudaPBOResource(ids[i]);
        if (res && res->get()) s_pboMap[ids[i]] = res; else delete res;
    }
    for (int i=0;i<count && !s_pboActive;++i){ auto it=s_pboMap.find(ids[i]); if(it!=s_pboMap.end()) s_pboActive=it->second; }
}
void unregisterAllPBOs() {
    for (auto &kv : s_pboMap) delete kv.second; s_pboMap.clear(); s_pboActive=nullptr;
}
void registerPBO(const Hermelin::GLBuffer& pbo) {
    ensureDeviceOnce();
    const GLuint id = pbo.id();
    auto it = s_pboMap.find(id);
    if (it == s_pboMap.end()) {
        auto* res = new bear_CudaPBOResource(id);
        if (res && res->get()) { s_pboMap[id]=res; if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[CUDA-Interop] auto-registered PBO id=%u", id); }
        else { delete res; LUCHS_LOG_HOST("[FATAL] failed to create CudaPBOResource id=%u", id); return; }
        it = s_pboMap.find(id);
    }
    s_pboActive = it->second;
}
void unregisterPBO() {  // nur aktives PBO lösen + Host-Pins/Event/Stream putzen
    if (s_hostRegEntropyPtr)  { cudaHostUnregister(s_hostRegEntropyPtr);  s_hostRegEntropyPtr=nullptr;  s_hostRegEntropyBytes=0; }
    if (s_hostRegContrastPtr) { cudaHostUnregister(s_hostRegContrastPtr); s_hostRegContrastPtr=nullptr; s_hostRegContrastBytes=0; }
    destroyEventsIfAny();
    if (s_copyStrm) { cudaStreamDestroy(s_copyStrm); s_copyStrm=nullptr; }
    delete s_pboActive; s_pboActive=nullptr;
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
    int tileSize, RendererState& state
){
#if !defined(__CUDA_ARCH__)
    const auto t0 = std::chrono::high_resolution_clock::now();
    double mapMs=0.0, mbMs=0.0, entMs=0.0, conMs=0.0;
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

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        ensureEventsOnce();
        if (s_evInit) cudaEventRecord(s_evStart, 0);
    }

    (void)cudaGetLastError();
    launch_mandelbrotHybrid(static_cast<uchar4*>(map.ptr),
                            static_cast<uint16_t*>(d_iterations.get()),
                            width,height, zoom, offset, maxIterations, tileSize);
    cudaError_t mbErrLaunch = cudaGetLastError();
    cudaError_t mbErrSync   = cudaSuccess;

    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        if (s_evInit) {
            cudaEventRecord(s_evStop, 0);
            mbErrSync = cudaEventSynchronize(s_evStop);
            float ms=0.0f;
            if (mbErrSync==cudaSuccess) {
                cudaEventElapsedTime(&ms, s_evStart, s_evStop);
            #if !defined(__CUDA_ARCH__)
                mbMs = ms;
            #endif
            }
        }
    }
    if (mbErrLaunch != cudaSuccess || mbErrSync != cudaSuccess)
        throw std::runtime_error("CUDA failure: mandelbrot kernel");

#if !defined(__CUDA_ARCH__)
    const auto tEC0 = std::chrono::high_resolution_clock::now();
#endif
    ::computeCudaEntropyContrast(
        static_cast<const uint16_t*>(d_iterations.get()),
        static_cast<float*>(d_entropy.get()),
        static_cast<float*>(d_contrast.get()),
        width, height, tileSize, maxIterations
    );
#if !defined(__CUDA_ARCH__)
    const auto tEC1 = std::chrono::high_resolution_clock::now();
    const double ecMs = std::chrono::duration<double, std::milli>(tEC1 - tEC0).count();
    entMs = ecMs * 0.5; conMs = ecMs * 0.5;
#endif

    // Host-Transfers (gepinnt, async)
    if (h_entropy.capacity()  < size_t(numTiles)) h_entropy.reserve(size_t(numTiles));
    if (h_contrast.capacity() < size_t(numTiles)) h_contrast.reserve(size_t(numTiles));
    ensureHostPinned(h_entropy,  s_hostRegEntropyPtr,  s_hostRegEntropyBytes);
    ensureHostPinned(h_contrast, s_hostRegContrastPtr, s_hostRegContrastBytes);
    h_entropy.resize(size_t(numTiles)); h_contrast.resize(size_t(numTiles));

    ensureCopyStreamOnce();
    CUDA_CHECK(cudaMemcpyAsync(h_entropy.data(),  d_entropy.get(),  enBytes, cudaMemcpyDeviceToHost, s_copyStrm));
    CUDA_CHECK(cudaMemcpyAsync(h_contrast.data(), d_contrast.get(), ctBytes, cudaMemcpyDeviceToHost, s_copyStrm));
    CUDA_CHECK(cudaStreamSynchronize(s_copyStrm));

    // Zoom-Entscheidung (hier neutral)
    shouldZoom = false; newOffset = offset;

#if !defined(__CUDA_ARCH__)
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.lastTimings.valid            = true;
    state.lastTimings.pboMap           = mapMs;
    state.lastTimings.mandelbrotTotal  = mbMs;
    state.lastTimings.mandelbrotLaunch = 0.0;
    state.lastTimings.mandelbrotSync   = 0.0;
    state.lastTimings.entropy          = entMs;
    state.lastTimings.contrast         = conMs;
    state.lastTimings.deviceLogFlush   = 0.0;

    if constexpr (Settings::performanceLogging)
        LUCHS_LOG_HOST("[PERF] path=gpu mp=%.2f mb=%.2f en=%.2f ct=%.2f tt=%.2f", mapMs, mbMs, entMs, conMs, totalMs);
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

bool verifyCudaGetErrorStringSafe() {
    const char* msg = cudaGetErrorString(cudaErrorInvalidValue);
    if (msg) { if constexpr (Settings::debugLogging) LUCHS_LOG_HOST("[CHECK] cudaGetErrorString(dummy)=\"%s\"", msg); return true; }
    LUCHS_LOG_HOST("[FATAL] cudaGetErrorString returned null"); return false;
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
