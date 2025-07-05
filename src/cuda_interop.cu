// Datei: src/cuda_interop.cu
// Zeilen: 352
// 🐭 Maus-Kommentar: Kolibri vollständig integriert: Adaptives Supersampling pro Tile.
// Flugente aktiv: float2 für maximale Performance. Panda-Modul (Entropie+Kontrast) vollständig erhalten.
// Schneefuchs: „Wer intelligent supersampelt, spart Performance für mehr Zoom.“
// Log bleibt ASCII-only.
// Kiwi: Reihenfolge der Heatmap-Berechnung korrigiert – Entropie/Kontrast jetzt nach dem Rendern ermittelt.

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

    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int numTiles = tilesX * tilesY;

    // --- DEBUG: Vorher Buffer auslesen
    if (Settings::debugLogging) {
        int dbg_before[3] = {-12345, -12345, -12345};
        CUDA_CHECK(cudaMemcpy(dbg_before, d_iterations, 3 * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[DEBUG] d_iterations BEFORE Kernel: [%d, %d, %d]\n", dbg_before[0], dbg_before[1], dbg_before[2]);
    }

    // 1. Starte Mandelbrot-Kernel (aktualisiert d_iterations)
    if (Settings::debugLogging) std::puts("[DEBUG] Mandelbrot-Kernel...");
    launch_mandelbrotHybrid(devPtr, d_iterations,
                            width, height,
                            zoom, offset,
                            maxIterations,
                            tileSize,
                            d_tileSupersampling);

    // --- DEBUG: Nachher Buffer auslesen
    if (Settings::debugLogging) {
        int dbg_after[3] = {-12345, -12345, -12345};
        CUDA_CHECK(cudaMemcpy(dbg_after, d_iterations, 3 * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[DEBUG] d_iterations AFTER Kernel: [%d, %d, %d]\n", dbg_after[0], dbg_after[1], dbg_after[2]);
    }

    // 2. Berechne Entropie und Kontrast per CUDA auf Basis aktueller Iterationen
    if (Settings::debugLogging) std::puts("[DEBUG] Entropy-Kernel...");
    computeCudaEntropyContrast(d_iterations, d_entropy, d_contrast,
                               width, height, tileSize, maxIterations);

    // 3. Kopiere Analyse-Daten auf Host
    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    // 4. Adaptive Supersampling-Stufen pro Tile setzen (Kolibri)
    h_tileSupersampling.resize(numTiles);
    for (int i = 0; i < numTiles; ++i) {
        h_tileSupersampling[i] = (h_entropy[i] > Settings::ENTROPY_THRESHOLD_HIGH) ? 4 :
                                 (h_entropy[i] > Settings::ENTROPY_THRESHOLD_LOW ) ? 2 : 1;
    }
    CUDA_CHECK(cudaMemcpy(d_tileSupersampling, h_tileSupersampling.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice));

    // --- DEBUG: Supersampling-Buffer loggen
    if (Settings::debugLogging && numTiles > 0) {
        std::printf("[SUPERSAMPLE] h_tileSupersampling[0]=%d [1]=%d [2]=%d\n",
                    h_tileSupersampling[0],
                    numTiles > 1 ? h_tileSupersampling[1] : -1,
                    numTiles > 2 ? h_tileSupersampling[2] : -1);
        std::vector<int> devCheck(numTiles);
        CUDA_CHECK(cudaMemcpy(devCheck.data(), d_tileSupersampling, numTiles * sizeof(int), cudaMemcpyHostToDevice));
        std::printf("[SUPERSAMPLE] d_tileSupersampling[0]=%d [1]=%d [2]=%d\n",
                    devCheck[0],
                    numTiles > 1 ? devCheck[1] : -1,
                    numTiles > 2 ? devCheck[2] : -1);
    }

    // 5. Auto-Zoom-Logik wie gehabt
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
