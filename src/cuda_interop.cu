// Datei: src/cuda_interop.cu
// Zeilen: 278
/* 🐭 Maus-Kommentar: CUDA-Interop mit kompaktem ASCII-Logging für Zoomanalyse.
   Jetzt mit dO (OffsetDist), dPx (Bildschirmpixel), Score, Entropie, Kontrast und Zielstatus – alles CSV-freundlich.
   Schneefuchs sieht klar: Kein Wildsprung bleibt unbemerkt.
   → Signatur fixiert: renderCudaFrame(...) mit double / double2 für präzise Tiefe. Linker-Link ist happy.
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
    int width,
    int height,
    double zoom,
    double2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    double2& newOffset,
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

    float zoom_f = static_cast<float>(zoom);
    float2 offset_f = make_float2(static_cast<float>(offset.x), static_cast<float>(offset.y));

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Launch MandelbrotKernel zoom %.2f maxIter %d supersampling %d\n", zoom, maxIterations, supersampling);
    }

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom_f, offset_f, maxIterations, supersampling);

    cudaDeviceSynchronize();  // sicherstellen, dass Fehler jetzt sichtbar sind
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] MandelbrotKernel launch failed: %s\n", cudaGetErrorString(kernelErr));
    }

    computeTileEntropy(d_iterations, d_entropy, width, height, tileSize, maxIterations);

    if (Settings::debugLogging) {
        std::puts("[DEBUG] PBO unmapped");
    }

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    h_entropy.resize(numTiles);
    CUDA_CHECK(cudaDeviceSynchronize());
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
            newOffset = result.newOffset;
            shouldZoom = result.shouldZoom;

            if (result.isNewTarget) {
                state.zoomResult.bestEntropy  = result.bestEntropy;
                state.zoomResult.bestContrast = result.bestContrast;
                state.zoomResult.bestIndex    = result.bestIndex;
            }
        }

        if (Settings::debugLogging) {
            if (result.bestIndex >= 0) {
                float minJump = Settings::MIN_JUMP_DISTANCE / zoom_f;
                std::printf(
                    "Zoom Z %.1e I %d E %.3f C %.3f S %.3f dO %.2e dPx %.1f minJ %.2e dE %.3f dC %.3f RelE %.2f RelC %.2f New %d\n",
                    zoom_f,
                    result.bestIndex,
                    result.bestEntropy,
                    result.bestContrast,
                    result.bestScore,
                    result.distance,
                    result.distance * zoom_f * width,
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

// 🐭 Maus-Kommentar: CSV-Log-Ausgabe aller Tile-Iterationen eines Frames – dient Analyse der Tiefenverteilung.
// Schneefuchs: Nur ASCII, keine GPU-Belastung, immer verständlich.
void logZoomEvaluation(const int* d_iterations, int width, int height, int tileSize, double zoom) {
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
