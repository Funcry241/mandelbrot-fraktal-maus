// Datei: src/cuda_interop.cu
// 🐭 Maus-Kommentar: Logging auf LUCHS_LOG_HOST umgestellt. Kein iostream mehr. Schneefuchs: deterministisch.

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"
#include <cuda_gl_interop.h>
#include <vector>
#include <cstdio>
#include <iomanip>
#include <sstream>

#ifndef __CUDA_ARCH__
  #include <chrono>
#endif

namespace CudaInterop {

static cudaGraphicsResource_t cudaPboResource = nullptr;
static bool pauseZoom = false;

void registerPBO(unsigned int pbo) {
    if (cudaPboResource) {
        LUCHS_LOG_HOST("[ERROR] registerPBO: already registered!");
        return;
    }

    LUCHS_LOG_HOST("[CU-INFO] registerPBO called with pbo=%u", pbo);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        LUCHS_LOG_HOST("[CU-ERROR] cudaGraphicsGLRegisterBuffer failed: %s", cudaGetErrorString(err));
    } else {
        LUCHS_LOG_HOST("[CU-INFO] cudaGraphicsGLRegisterBuffer succeeded");
    }
}

void unregisterPBO() {
    if (cudaPboResource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPboResource));
        cudaPboResource = nullptr;
    }
}

void renderCudaFrame(
    int* d_iterations, float* d_entropy, float* d_contrast,
    int width, int height, float zoom, float2 offset, int maxIterations,
    std::vector<float>& h_entropy, std::vector<float>& h_contrast,
    float2& newOffset, bool& shouldZoom, int tileSize,
    RendererState& state
) {
    if (!cudaPboResource)
        throw std::runtime_error("[FATAL] CUDA PBO not registered!");

#ifndef __CUDA_ARCH__
    const auto t0 = std::chrono::high_resolution_clock::now();
#endif

    const int totalPixels = width * height;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    if (Settings::debugLogging) {
        CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_entropy, 0, numTiles * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_contrast, 0, numTiles * sizeof(float)));
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CU-FRAME] zoom=%.5f offset=(%.5f %.5f) iter=%d tile=%d",
                       zoom, offset.x, offset.y, maxIterations, tileSize);
    }

    if (Settings::debugLogging) {
        int dbg_before[3]{};
        CUDA_CHECK(cudaMemcpy(dbg_before, d_iterations, sizeof(dbg_before), cudaMemcpyDeviceToHost));
        launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, tileSize);
        int dbg_after[3]{};
        CUDA_CHECK(cudaMemcpy(dbg_after, d_iterations, sizeof(dbg_after), cudaMemcpyDeviceToHost));
        LUCHS_LOG_HOST("[CU-KERNEL] iters: %d->%d | %d->%d | %d->%d",
                       dbg_before[0], dbg_after[0],
                       dbg_before[1], dbg_after[1],
                       dbg_before[2], dbg_after[2]);
    } else {
        launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, tileSize);
    }

    computeCudaEntropyContrast(d_iterations, d_entropy, d_contrast, width, height, tileSize, maxIterations);

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
                LUCHS_LOG_HOST("[CU-ZOOM] idx=%d entropy=%.3f contrast=%.3f -> (%.5f %.5f) new=%d zoom=%d",
                               result.bestIndex,
                               result.bestEntropy,
                               result.bestContrast,
                               result.newOffset.x, result.newOffset.y,
                               result.isNewTarget ? 1 : 0,
                               result.shouldZoom ? 1 : 0);
            }
        } else if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CU-ZOOM] No suitable target");
        }
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));

#ifndef __CUDA_ARCH__
    const auto t1 = std::chrono::high_resolution_clock::now();
    const float totalMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[CU-PERF] total=%.2f ms", totalMs);
#endif
}

void setPauseZoom(bool pause) { pauseZoom = pause; }
[[nodiscard]] bool getPauseZoom() { return pauseZoom; }

} // namespace CudaInterop
