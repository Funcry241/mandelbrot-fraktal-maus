// Datei: src/cuda_interop.cu
// Zeilen: 316
/* 🐭 Maus-Kommentar: CUDA-Interop mit Entropie und Kontrast für Heatmap und Auto-Zoom.
   Flugente: Alle Koordinaten sind jetzt float2! Kein double2 mehr im Spiel.
   Panda: Entropie+Kontrast-Analyse bleibt vollständig erhalten.
   Schneefuchs: „Wer float kann, braucht kein double – solange keine Galaxie explodiert.“
   Log bleibt ASCII-only.
*/

#include "pch.hpp"
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"
#include <vector>
#include <cstdio>

namespace CudaInterop {

static cudaGraphicsResource_t cudaPboResource = nullptr;
static bool pauseZoom = false;

void registerPBO(unsigned int pbo) {
    if (cudaPboResource != nullptr) {
        std::cerr << "[ERROR] registerPBO called but resource is already registered!\n";
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
    int supersampling,
    RendererState& state
) {
    if (!cudaPboResource) {
        throw std::runtime_error("[FATAL] CUDA PBO not registered before renderCudaFrame.");
    }

    if (Settings::debugLogging) {
        std::printf("[Zoom] Auto-Zoom is %s\n", pauseZoom ? "PAUSED" : "ACTIVE");
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    if (Settings::debugLogging) {
        std::printf("[DEBUG] PBO mapped: %p (size = %zu)\n", (void*)devPtr, size);
    }

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Launch MandelbrotKernel zoom %.2f maxIter %d supersampling %d\n", zoom, maxIterations, supersampling);
    }

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations, supersampling);

    cudaDeviceSynchronize();
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] MandelbrotKernel launch failed: %s\n", cudaGetErrorString(kernelErr));
    }

    computeEntropyContrast(d_iterations, d_entropy, d_contrast, width, height, tileSize, maxIterations);

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contrast.data(), d_contrast, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    shouldZoom = false;

    if (!pauseZoom) {
        // Volle Flugente: float2 überall!
        ZoomLogic::ZoomResult result = ZoomLogic::evaluateZoomTarget(
            h_entropy,
            h_contrast,
            offset,
            zoom,
            width,
            height,
            tileSize,
            state.offset,
            state.zoomResult.bestIndex,
            state.zoomResult.bestEntropy,
            state.zoomResult.bestContrast
        );

        if (result.bestIndex >= 0) {
            newOffset = result.newOffset;   // float2 zu float2 – keine Umwandlung
            shouldZoom = result.shouldZoom;

            if (result.isNewTarget) {
                state.zoomResult.bestEntropy  = result.bestEntropy;
                state.zoomResult.bestContrast = result.bestContrast;
                state.zoomResult.bestIndex    = result.bestIndex;
            }
        }

        if (Settings::debugLogging) {
            if (result.bestIndex >= 0) {
                float minJump = Settings::MIN_JUMP_DISTANCE / zoom;
                std::printf(
                    "Zoom Z %.1e I %d E %.3f C %.3f S %.3f dO %.2e dPx %.1f minJ %.2e dE %.3f dC %.3f RelE %.2f RelC %.2f New %d\n",
                    zoom,
                    result.bestIndex,
                    result.bestEntropy,
                    result.bestContrast,
                    result.bestScore,
                    result.distance,
                    result.distance * zoom * width,
                    minJump,
                    result.relEntropyGain,
                    result.relContrastGain,
                    result.relEntropyGain,
                    result.relContrastGain,
                    result.isNewTarget ? 1 : 0
                );
            } else {
                float avgEntropy = 0.0f;
                int countAbove = 0;
                for (float h : h_entropy) {
                    avgEntropy += h;
                    if (h > Settings::VARIANCE_THRESHOLD) countAbove++;
                }
                avgEntropy /= h_entropy.size();
                std::printf("Zoom NoZoom TilesAbove %d AvgEntropy %.5f\n", countAbove, avgEntropy);
            }
        }

        if (!result.isNewTarget) {
            state.zoomResult = result;
        }
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
}

void setPauseZoom(bool pause) {
    pauseZoom = pause;
}

bool getPauseZoom() {
    return pauseZoom;
}

void logZoomEvaluation(const int* d_iterations, int width, int height, int tileSize, float zoom) {
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;

    std::vector<int> h_iterations(width * height);
    cudaMemcpy(h_iterations.data(), d_iterations, sizeof(int) * width * height, cudaMemcpyDeviceToHost);

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int sum = 0;
            int count = 0;

            for (int dy = 0; dy < tileSize; ++dy) {
                for (int dx = 0; dx < tileSize; ++dx) {
                    int x = tx * tileSize + dx;
                    int y = ty * tileSize + dy;
                    if (x >= width || y >= height) continue;
                    sum += h_iterations[y * width + x];
                    ++count;
                }
            }

            float avg = (count > 0) ? (float)sum / count : 0.0f;
            std::printf("[ZoomEvalCSV] %d,%d,%.4f,%.2f\n", tx, ty, zoom, avg);
        }
    }
}

} // namespace CudaInterop
