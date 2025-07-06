// Datei: src/cuda_interop.cu
// Zeilen: 208
// 🐭 Maus-Kommentar: Kolibri+Kiwi+Otter – Buffer-Nullung, Host/Device-Sync und Logging maximal kompakt, robust und ASCII-clean. Keine OOBs, kein Shadowing, 100% deterministisch. Schneefuchs approved.
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

    int totalPixels = width * height;
    int tilesX = (width + tileSize - 1) / tileSize, tilesY = (height + tileSize - 1) / tileSize, numTiles = tilesX * tilesY;
    CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_entropy, 0, numTiles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_contrast, 0, numTiles * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_tileSupersampling, 0, numTiles * sizeof(int)));

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr; size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    if (Settings::debugLogging) {
        std::printf("[PTRS] devPtr=%p d_iter=%p d_ent=%p d_con=%p d_sup=%p\n", devPtr, d_iterations, d_entropy, d_contrast, d_tileSupersampling);
        int dbg_before[3]{-12345}; CUDA_CHECK(cudaMemcpy(dbg_before, d_iterations, 3 * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[DEBUG] d_iterations BEFORE Kernel: [%d, %d, %d]\n", dbg_before[0], dbg_before[1], dbg_before[2]);
    }

    if (Settings::debugLogging) std::puts("[DEBUG] Mandelbrot-Kernel...");
    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, tileSize, d_tileSupersampling, supersampling);

    if (Settings::debugLogging) {
        int dbg_after[3]{-12345}; CUDA_CHECK(cudaMemcpy(dbg_after, d_iterations, 3 * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[DEBUG] d_iterations AFTER Kernel: [%d, %d, %d]", dbg_after[0], dbg_after[1], dbg_after[2]);
        for (int i = 0; i < 3; ++i) if (dbg_after[i] < 0) std::printf(" [WARN] Found <0 value! Check buffer init or kernel OOB.\n");
        std::puts("");
    }

    if (Settings::debugLogging) std::puts("[DEBUG] Entropy-Kernel...");
    computeCudaEntropyContrast(d_iterations, d_entropy, d_contrast, width, height, tileSize, maxIterations);

    h_entropy.resize(numTiles); h_contrast.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    // Adaptive Supersampling pro Tile (Kolibri)
    h_tileSupersampling.resize(numTiles);
    for (int i = 0; i < numTiles; ++i)
        h_tileSupersampling[i] = (h_entropy[i] > Settings::ENTROPY_THRESHOLD_HIGH) ? 4 :
                                 (h_entropy[i] > Settings::ENTROPY_THRESHOLD_LOW)  ? 2 : 1;
    CUDA_CHECK(cudaMemcpy(d_tileSupersampling, h_tileSupersampling.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice));

    if (Settings::debugLogging && numTiles > 0) {
        std::printf("[SUPERSAMPLE] h_tileSupersampling[0]=%d [1]=%d [2]=%d\n",
            h_tileSupersampling[0], numTiles > 1 ? h_tileSupersampling[1] : -1, numTiles > 2 ? h_tileSupersampling[2] : -1);
        std::vector<int> devCheck(numTiles);
        CUDA_CHECK(cudaMemcpy(devCheck.data(), d_tileSupersampling, numTiles * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[SUPERSAMPLE] d_tileSupersampling[0]=%d [1]=%d [2]=%d\n",
            devCheck[0], numTiles > 1 ? devCheck[1] : -1, numTiles > 2 ? devCheck[2] : -1);
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
            if (result.isNewTarget) state.zoomResult = result;
        }
    }
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
}

void setPauseZoom(bool pause) { pauseZoom = pause; }
bool getPauseZoom() { return pauseZoom; }

} // namespace CudaInterop
