// Datei: src/cuda_interop.cu
// Zeilen: 178
// 🐭 Maus-Kommentar: CUDA-Interop delegiert Zielanalyse jetzt an ZoomLogic. Kompakter, modularer, klarer. Schneefuchs: „Nur wer delegiert, bleibt flexibel.“

#include "pch.hpp"  // 💡 Muss als erstes stehen!
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"  // 🔥 Overlay-Toggle per Taste

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

void logZoomEvaluation(const int* d_iterations, int width, int height, int maxIterations, double zoom) {
#if ENABLE_ZOOM_LOGGING
    std::vector<int> h_iters(width * height);
    CUDA_CHECK(cudaMemcpy(h_iters.data(), d_iterations, h_iters.size() * sizeof(int), cudaMemcpyDeviceToHost));

    double sum = 0.0, sumSq = 0.0;
    int minIt = maxIterations;
    int maxIt = 0;
    int escapeCount = 0;

    for (int it : h_iters) {
        sum += it;
        sumSq += it * it;
        if (it < minIt) minIt = it;
        if (it > maxIt) maxIt = it;
        if (it < 5) escapeCount++;
    }

    const int total = static_cast<int>(h_iters.size());
    const double mean = sum / total;
    const double variance = (sumSq / total) - (mean * mean);
    const double escapeRatio = static_cast<double>(escapeCount) / total;

    bool valid = (escapeRatio < 0.98) && (variance > 0.05) && (mean > 5.0);

    std::printf("ZoomEval Z %.1e MeanIt %.2f VarIt %.2f Escape %.3f Min %d Max %d Valid %d\n",
        zoom, mean, variance, escapeRatio, minIt, maxIt, valid ? 1 : 0);
#endif
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
    double2& newOffset,
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
            h_entropy,
            offset,
            zoom_f,
            width,
            height,
            tileSize,
            make_float2(static_cast<float>(state.offset.x), static_cast<float>(state.offset.y)),
            state.zoomResult.bestIndex,
            state.zoomResult.bestEntropy,
            state.zoomResult.bestContrast
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
                result.bestEntropy - state.zoomResult.bestEntropy,
                result.bestContrast - state.zoomResult.bestContrast,
                result.relEntropyGain, result.relContrastGain,
                (result.bestIndex != state.zoomResult.bestIndex) ? 1 : 0,
                result.isNewTarget ? 1 : 0
            );
            logZoomEvaluation(d_iterations, width, height, maxIterations, zoom);
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

        state.zoomResult = result;  // 🔁 ZoomResult speichern (auch Kontrast-Heatmap!)
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
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_P || key == GLFW_KEY_SPACE) {
            pauseZoom = !pauseZoom;
            std::cout << "[Zoom] Auto-Zoom " << (pauseZoom ? "paused" : "resumed") << "\n";
        }

        if (key == GLFW_KEY_H) {
            HeatmapOverlay::toggle();
            if (Settings::debugLogging) {
                std::puts("[DEBUG] Heatmap overlay toggled (H)");
            }
        }
    }
}

} // namespace CudaInterop
