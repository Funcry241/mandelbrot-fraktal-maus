// Datei: src/cuda_interop.cu
// Zeilen: 137
// 🐭 Maus-Kommentar: CUDA-Interop delegiert Zielanalyse jetzt an ZoomLogic. Kompakter, modularer, klarer. Schneefuchs: „Nur wer delegiert, bleibt flexibel.“

#include "pch.hpp"  // 💡 Muss als erstes stehen!
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"

#define ENABLE_ZOOM_LOGGING 1  // Set to 0 to disable local zoom analysis logs

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
    int width,
    int height,
    double zoom,
    double2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize,
    RendererState& state
) {
    if (!cudaPboResource) {
        throw std::runtime_error("[FATAL] CUDA PBO not registered before renderCudaFrame.");
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    float zoom_f = static_cast<float>(zoom);
    float2 offset_f = make_float2(static_cast<float>(offset.x), static_cast<float>(offset.y));

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom_f, offset_f, maxIterations);
    computeTileEntropy(d_iterations, d_entropy, width, height, tileSize, maxIterations);

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    h_entropy.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    shouldZoom = false;

    if (!pauseZoom) {
        ZoomLogic::ZoomResult result = ZoomLogic::evaluateZoomTarget(
            h_entropy, offset_f, zoom_f, width, height, tileSize, state
        );

        if (result.bestIndex >= 0) {
            shouldZoom = result.shouldZoom;
            newOffset = result.newOffset;
        }

#if ENABLE_ZOOM_LOGGING
        if (shouldZoom) {
            std::printf(
                "ZoomLog Z %.5e Idx %d Ent %.5f S %.5f Dist %.6f Min %.6f dE %.4f dC %.4f RelE %.3f RelC %.3f dI %d New %d\n",
                zoom_f, result.bestIndex, result.bestEntropy, result.bestScore, result.distance, result.minDistance,
                result.bestEntropy - state.lastEntropy,
                result.bestContrast - state.lastContrast,
                result.relEntropyGain, result.relContrastGain,
                (result.bestIndex != state.lastIndex) ? 1 : 0,
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
            std::printf("ZoomLog NoZoom TilesAbove %d AvgEntropy %.5f\n", countAbove, avgEntropy);
        }
#endif

        state.lastIndex = result.bestIndex;
        state.lastEntropy = result.bestEntropy;
        state.lastContrast = result.bestContrast;
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
}

void setPauseZoom(bool pause) {
    pauseZoom = pause;
}

bool getPauseZoom() {
    return pauseZoom;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS && (key == GLFW_KEY_P || key == GLFW_KEY_SPACE)) {
        pauseZoom = !pauseZoom;
        std::cout << "[Zoom] Auto-Zoom " << (pauseZoom ? "paused" : "resumed") << "\n";
    }
}

} // namespace CudaInterop
