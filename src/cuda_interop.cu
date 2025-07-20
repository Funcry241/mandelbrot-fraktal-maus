// Datei: src/cuda_interop.cu
// Zeilen: 258
// 🐭 Maus-Kommentar: Alpha 49b – Logging-Reduktion nach „Gepard 2.0“: Debug-Ausgaben sind nun ASCII-clean, einzeilig, kompakt. Vorher/Nachher-Vergleiche der Iterationen, Supersampling-Check und Performance-Log in einer Zeile. Schneefuchs sagt: „Prägnanz ist Präzision.“
#include "pch.hpp"
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

#ifndef __CUDA_ARCH__
  #include <chrono>
#endif

namespace CudaInterop {

static cudaGraphicsResource_t cudaPboResource = nullptr;
static bool pauseZoom = false;

void registerPBO(unsigned int pbo) {
    if (cudaPboResource) {
        std::fprintf(stderr, "[ERROR] registerPBO: already registered!\n");
        return;
    }
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
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
    float2& newOffset, bool& shouldZoom, int tileSize, int supersampling,
    RendererState& state, int* d_tileSupersampling, std::vector<int>& h_tileSupersampling
) {
    if (!cudaPboResource)
        throw std::runtime_error("[FATAL] CUDA PBO not registered!");

#ifndef __CUDA_ARCH__
    auto t0 = std::chrono::high_resolution_clock::now();
#endif

    int totalPixels = width * height;
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int numTiles = tilesX * tilesY;

    CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_entropy, 0, numTiles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_contrast, 0, numTiles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_tileSupersampling, 0, numTiles * sizeof(int)));

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr; size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    if (Settings::debugLogging) {
        std::printf("[FRAME] zoom=%.6g offset=(%.6g, %.6g) maxIter=%d tileSize=%d supersampling=%d\n",
                    zoom, offset.x, offset.y, maxIterations, tileSize, supersampling);
        std::printf("[PTRS] devPtr=%p d_iter=%p d_ent=%p d_con=%p d_sup=%p\n",
                    devPtr, d_iterations, d_entropy, d_contrast, d_tileSupersampling);
    }

    int dbg_before[3]{-12345}, dbg_after[3]{-12345};
    if (Settings::debugLogging) {
        CUDA_CHECK(cudaMemcpy(dbg_before, d_iterations, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    }

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, tileSize, d_tileSupersampling, supersampling);

    if (Settings::debugLogging) {
        CUDA_CHECK(cudaMemcpy(dbg_after, d_iterations, 3 * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[DEBUG] Kernel iterations: %d→%d %d→%d %d→%d\n",
            dbg_before[0], dbg_after[0],
            dbg_before[1], dbg_after[1],
            dbg_before[2], dbg_after[2]);
    }

    if (Settings::debugLogging) std::puts("[DEBUG] Entropy+Contrast launched");

    computeCudaEntropyContrast(d_iterations, d_entropy, d_contrast, width, height, tileSize, maxIterations);

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    h_tileSupersampling.resize(numTiles);
    for (int i = 0; i < numTiles; ++i)
        h_tileSupersampling[i] = (h_entropy[i] > Settings::ENTROPY_THRESHOLD_HIGH) ? 4 :
                                 (h_entropy[i] > Settings::ENTROPY_THRESHOLD_LOW)  ? 2 : 1;
    CUDA_CHECK(cudaMemcpy(d_tileSupersampling, h_tileSupersampling.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice));

    if (Settings::debugLogging && numTiles > 2) {
        std::vector<int> devCheck(numTiles);
        CUDA_CHECK(cudaMemcpy(devCheck.data(), d_tileSupersampling, numTiles * sizeof(int), cudaMemcpyDeviceToHost));
        bool match = devCheck[0] == h_tileSupersampling[0] &&
                     devCheck[1] == h_tileSupersampling[1] &&
                     devCheck[2] == h_tileSupersampling[2];

        if (match) {
            std::printf("[DEBUG] Supersampling Tile[0-2]: %d %d %d\n",
                h_tileSupersampling[0], h_tileSupersampling[1], h_tileSupersampling[2]);
        } else {
            std::printf("[DEBUG] Supersample mismatch – host: %d %d %d | device: %d %d %d\n",
                h_tileSupersampling[0], h_tileSupersampling[1], h_tileSupersampling[2],
                devCheck[0], devCheck[1], devCheck[2]);
        }
    }

    shouldZoom = false;
    if (!pauseZoom) {
        auto result = ZoomLogic::evaluateZoomTarget(
            h_entropy, h_contrast, offset, zoom, width, height, tileSize,
            state.offset, state.zoomResult.bestIndex, state.zoomResult.bestEntropy, state.zoomResult.bestContrast
        );
        if (result.bestIndex >= 0) {
            newOffset = result.newOffset;
            shouldZoom = result.shouldZoom;
            state.zoomResult = result;

            if (Settings::debugLogging) {
                std::printf("[ZOOM] Target: idx=%d entropy=%.3f contrast=%.3f -> offset=(%.6g, %.6g) new=%d zoom=%d\n",
                    result.bestIndex, result.bestEntropy, result.bestContrast,
                    newOffset.x, newOffset.y, result.isNewTarget ? 1 : 0, result.shouldZoom ? 1 : 0);
            }
        } else if (Settings::debugLogging) {
            std::puts("[ZOOM] No suitable target found.");
        }
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));

#ifndef __CUDA_ARCH__
    auto t1 = std::chrono::high_resolution_clock::now();
    float totalMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    if (Settings::debugLogging)
        std::printf("[Perf] cuda_interop total=%.2fms\n", totalMs);
#endif
}

void setPauseZoom(bool pause) { pauseZoom = pause; }
[[nodiscard]] bool getPauseZoom() { return pauseZoom; }

} // namespace CudaInterop
