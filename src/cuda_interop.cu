// Datei: src/cuda_interop.cu
// Zeilen: 336
/* 🐭 Maus-Kommentar: Kolibri vollständig integriert: Adaptives Supersampling pro Tile.
   Flugente aktiv: float2 für maximale Performance. Panda-Modul (Entropie+Kontrast) vollständig erhalten.
   Schneefuchs: „Wer intelligent supersampelt, spart Performance für mehr Zoom.“
   Log bleibt ASCII-only.
*/

#include "pch.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"       // Deklaration von launch_mandelbrotHybrid, computeCudaEntropyContrast
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
    if (cudaPboResource != nullptr) {
        std::fprintf(stderr, "[ERROR] registerPBO called but resource is already registered!\n");
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
    int /*supersampling*/, // globaler Fallback, intern ignoriert
    RendererState& state,
    int* d_tileSupersampling,
    std::vector<int>& h_tileSupersampling
) {
    if (!cudaPboResource) {
        throw std::runtime_error("[FATAL] CUDA PBO not registered before renderCudaFrame.");
    }

    // Map OpenGL PBO to CUDA pointer
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    // Berechne Entropie und Kontrast per CUDA
    computeCudaEntropyContrast(d_iterations, d_entropy, d_contrast,
                               width, height, tileSize, maxIterations);

    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int numTiles = tilesX * tilesY;

    // Kopiere Analyse-Daten auf Host
    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    // 🦜 Kolibri: Adaptive Supersampling-Stufen pro Tile setzen
    h_tileSupersampling.resize(numTiles);
    for (int i = 0; i < numTiles; ++i) {
        h_tileSupersampling[i] = (h_entropy[i] > Settings::ENTROPY_THRESHOLD_HIGH) ? 4 :
                                 (h_entropy[i] > Settings::ENTROPY_THRESHOLD_LOW ) ? 2 : 1;
    }
    CUDA_CHECK(cudaMemcpy(d_tileSupersampling, h_tileSupersampling.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice));

    // Starte Mandelbrot-Kernel mit adaptivem Supersampling
    launch_mandelbrotHybrid(devPtr, d_iterations,
                            width, height,
                            zoom, offset,
                            maxIterations,
                            tileSize,
                            d_tileSupersampling);

    // Auto-Zoom-Logik
    shouldZoom = false;
    if (!pauseZoom) {
        ZoomLogic::ZoomResult result = ZoomLogic::evaluateZoomTarget(
            h_entropy, h_contrast,
            offset, zoom,
            width, height,
            tileSize,
            state.offset,
            state.zoomResult.bestIndex,
            state.zoomResult.bestEntropy,
            state.zoomResult.bestContrast
        );
        if (result.bestIndex >= 0) {
            newOffset = result.newOffset;
            shouldZoom = result.shouldZoom;
            if (result.isNewTarget) state.zoomResult = result;
        }
    }

    // Unmap PBO
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
}

void setPauseZoom(bool pause) {
    pauseZoom = pause;
}

bool getPauseZoom() {
    return pauseZoom;
}

} // namespace CudaInterop
