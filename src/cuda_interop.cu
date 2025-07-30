// Datei: src/cuda_interop.cu
// 🐜 Schwarze Ameise: Klare Parametrisierung, deterministisches Logging, robustes Ressourcenhandling.
// 🦦 Otter: Explizite und einheitliche Übergabe aller Parameter. Fehler- und Kontextlogging überall.
// 🦊 Schneefuchs: Keine impliziten Zugriffe, transparente Speicher- und Fehlerprüfung.

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include <cuda_gl_interop.h>
#include <vector>

#ifndef CUDA_ARCH
#include <chrono>
#endif

namespace CudaInterop {

static cudaGraphicsResource_t cudaPboResource = nullptr;
static bool pauseZoom = false;
static bool luchsBabyInitDone = false;

void logCudaDeviceContext(const char* context) {
int device = -1;
cudaError_t err = cudaGetDevice(&device);
LUCHS_LOG_HOST("[CTX] %s: cudaGetDevice() = %d (%s)", context, device, cudaGetErrorString(err));
}

void registerPBO(unsigned int pbo) {
if (cudaPboResource) {
LUCHS_LOG_HOST("[ERROR] registerPBO: already registered!");
return;
}

GLint boundBefore = 0;
glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBefore);
LUCHS_LOG_HOST("[CHECK] GL bind state BEFORE bind: %d", boundBefore);

glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

GLint boundAfter = 0;
glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundAfter);
LUCHS_LOG_HOST("[CHECK] GL bind state AFTER  bind: %d (expected: %u)", boundAfter, pbo);

if (boundAfter != static_cast<GLint>(pbo)) {
    LUCHS_LOG_HOST("[FATAL] GL bind failed - buffer %u was not bound (GL reports: %d)", pbo, boundAfter);
    throw std::runtime_error("glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo) failed - buffer not active");
}

if (Settings::debugLogging)
    LUCHS_LOG_HOST("[CU-PBO] Preparing to register PBO ID %u", pbo);

cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
if (err != cudaSuccess) {
    LUCHS_LOG_HOST("[CU-PBO] cudaGraphicsGLRegisterBuffer FAILED: %s", cudaGetErrorString(err));
    throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed");
}

if (Settings::debugLogging) {
    LUCHS_LOG_HOST("[CU-PBO] Registered GL buffer ID %u -> cudaPboResource: %p", pbo, (void*)cudaPboResource);
}

logCudaDeviceContext("after registerPBO");

}

void renderCudaFrame(
int* d_iterations,
float* d_entropy,
float* d_contrast,
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
if (Settings::debugLogging)
LUCHS_LOG_HOST("[ENTER] renderCudaFrame()");

logCudaDeviceContext("renderCudaFrame ENTER");

if (!cudaPboResource)
    throw std::runtime_error("[FATAL] CUDA PBO not registered!");

#ifndef CUDA_ARCH
const auto t0 = std::chrono::high_resolution_clock::now();
#endif

const int totalPixels = width * height;
const int tilesX = (width + tileSize - 1) / tileSize;
const int tilesY = (height + tileSize - 1) / tileSize;
const int numTiles = tilesX * tilesY;

CUDA_CHECK(cudaSetDevice(0));

// Debug: Pointer-Attribute prüfen
cudaPointerAttributes attrCheck;
cudaError_t attrErr = cudaPointerGetAttributes(&attrCheck, d_iterations);
LUCHS_LOG_HOST("[DEBUG] attrCheck result: err=%d, type=%d, device=%d, hostPtr=%p, devicePtr=%p",
    (int)attrErr, (int)attrCheck.type, (int)attrCheck.device,
    (void*)attrCheck.hostPointer, (void*)attrCheck.devicePointer);

// Speicher initialisieren
cudaError_t err = cudaMemset(d_iterations, 0, totalPixels * sizeof(int));
LUCHS_LOG_HOST("[MEM] d_iterations memset: %d pixels -> %zu bytes", totalPixels, totalPixels * sizeof(int));
if (err != cudaSuccess) throw std::runtime_error("cudaMemset d_iterations failed");

CUDA_CHECK(cudaMemset(d_entropy, 0, numTiles * sizeof(float)));
CUDA_CHECK(cudaMemset(d_contrast, 0, numTiles * sizeof(float)));

if (Settings::debugLogging)
    LUCHS_LOG_HOST("[MAP] Mapping CUDA-GL resource %p", (void*)cudaPboResource);

CUDA_CHECK(cudaDeviceSynchronize());
CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));

uchar4* devPtr = nullptr;
size_t size = 0;
CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

if (Settings::debugLogging)
    LUCHS_LOG_HOST("[MAP] Mapped pointer: %p (%zu bytes)", (void*)devPtr, size);

if (!devPtr) {
    LUCHS_LOG_HOST("[FATAL] Kernel skipped: surface pointer is null");
    return;
}

// Einmalige Initialisierung Device-Logbuffer
if (!luchsBabyInitDone) {
    LuchsLogger::initCudaLogBuffer(0);
    luchsBabyInitDone = true;
}

if (Settings::debugLogging) {
    int dbg_before[3]{};
    CUDA_CHECK(cudaMemcpy(dbg_before, d_iterations, sizeof(dbg_before), cudaMemcpyDeviceToHost));

    LUCHS_LOG_HOST("[KERNEL] launch_mandelbrotHybrid(surface=%p, w=%d, h=%d, zoom=%.5f, offset=(%.5f %.5f), iter=%d, tile=%d)",
        (void*)devPtr, width, height, zoom, offset.x, offset.y, maxIterations, tileSize);

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, tileSize);
    LuchsLogger::flushDeviceLogToHost();

    LUCHS_LOG_HOST("[KERNEL] mandelbrotKernel(...) launched");

    int dbg_after[3]{};
    CUDA_CHECK(cudaMemcpy(dbg_after, d_iterations, sizeof(dbg_after), cudaMemcpyDeviceToHost));
    LUCHS_LOG_HOST("[KERNEL] iters changed: %d->%d | %d->%d | %d->%d",
                   dbg_before[0], dbg_after[0],
                   dbg_before[1], dbg_after[1],
                   dbg_before[2], dbg_after[2]);
} else {
    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, tileSize);
    LuchsLogger::flushDeviceLogToHost();
}

::computeCudaEntropyContrast(d_iterations, d_entropy, d_contrast, width, height, tileSize, maxIterations);

h_entropy.resize(numTiles);
h_contrast.resize(numTiles);
CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

shouldZoom = false;
if (!pauseZoom) {
    const auto result = ZoomLogic::evaluateZoomTarget(
        h_entropy, h_contrast, offset, zoom, width, height, tileSize,
        state.offset, state.zoomResult.bestIndex, state.zoomResult.bestEntropy, state.zoomResult.bestContrast
    );

    if (result.bestIndex >= 0) {
        newOffset = result.newOffset;
        shouldZoom = result.shouldZoom;
        state.zoomResult = result;

        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM] idx=%d entropy=%.3f contrast=%.3f -> (%.5f %.5f) new=%d zoom=%d",
                           result.bestIndex,
                           result.bestEntropy,
                           result.bestContrast,
                           result.newOffset.x, result.newOffset.y,
                           result.isNewTarget ? 1 : 0,
                           result.shouldZoom ? 1 : 0);
        }
    } else if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOM] No suitable target");
    }
}

CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));

if (Settings::debugLogging) {
    LUCHS_LOG_HOST("[UNMAP] PBO unmapped successfully");
    LUCHS_LOG_HOST("[KERNEL] renderCudaFrame finished");
}

#ifndef CUDA_ARCH
const auto t1 = std::chrono::high_resolution_clock::now();
const float totalMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
if (Settings::debugLogging)
LUCHS_LOG_HOST("[PERF] renderCudaFrame() = %.2f ms", totalMs);
#endif
}

void setPauseZoom(bool pause) { pauseZoom = pause; }
[[nodiscard]] bool getPauseZoom() { return pauseZoom; }

bool precheckCudaRuntime() {
int deviceCount = 0;
cudaError_t err1 = cudaFree(0);
cudaError_t err2 = cudaGetDeviceCount(&deviceCount);

LUCHS_LOG_HOST("[CUDA] precheck err1=%d err2=%d count=%d", (int)err1, (int)err2, deviceCount);
return (err1 == cudaSuccess && err2 == cudaSuccess && deviceCount > 0);

}

bool verifyCudaGetErrorStringSafe() {
cudaError_t dummy = cudaErrorInvalidValue;
const char* msg = cudaGetErrorString(dummy);

if (msg) {
    LUCHS_LOG_HOST("[CHECK] cudaGetErrorString(dummy) = \"%s\"", msg);
    LUCHS_LOG_HOST("[PASS] Host-seitige Fehleraufloesung funktioniert gefahrlos");
    return true;
} else {
    LUCHS_LOG_HOST("[FATAL] cudaGetErrorString returned null - das riecht nach Treibergift");
    return false;
}

}

void unregisterPBO() {
if (cudaPboResource) {
cudaGraphicsUnregisterResource(cudaPboResource);
cudaPboResource = nullptr;
if (Settings::debugLogging)
LUCHS_LOG_HOST("[CU-PBO] Unregistered PBO resource");
}
}

} // namespace CudaInterop
