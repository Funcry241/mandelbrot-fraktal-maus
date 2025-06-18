// Datei: src/cuda_interop.cu
// Zeilen: 151
// 🐅 Maus-Kommentar: CUDA/OpenGL-Interop für PBO-Mapping & Fraktalberechnung. Auto-Zoom via Entropieanalyse (pro Tile), bestes Tile wird ermittelt, Zoom-Ziel korrekt in Fraktalkoordinaten umgerechnet (inkl. Mittelpunktabzug wie im Kernel). Jetzt mit "SmoothZoom + NearbyBias" zur Vermeidung von Flattern. Schneefuchs hätte gesagt: „Sanft wie ein Gleitflug zum Entropie-Hotspot.“

#include "pch.hpp"  // 💡 Muss als erstes stehen!

#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"

namespace CudaInterop {

static cudaGraphicsResource_t cudaPboResource;
static bool pauseZoom = false;

void registerPBO(unsigned int pbo) {
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
    float zoom,
    float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize
) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations);
    computeTileEntropy(d_iterations, d_entropy, width, height, tileSize, maxIterations);

    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int numTiles = tilesX * tilesY;
    h_entropy.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    if (!pauseZoom) {
        int bestIndex = -1;
        float bestScore = -1.0f;

        for (int i = 0; i < numTiles; ++i) {
            int bx = i % tilesX;
            int by = i / tilesX;

            float centerX = (bx + 0.5f) * tileSize;
            float centerY = (by + 0.5f) * tileSize;

            float2 tileCenter = {
                (centerX - width  / 2.0f) / zoom + offset.x,
                (centerY - height / 2.0f) / zoom + offset.y
            };

            float2 delta = {
                tileCenter.x - offset.x,
                tileCenter.y - offset.y
            };

            float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
            float score = h_entropy[i] / (1.0f + 50.0f * dist); // 📊 Entropie-Bonus für nähere Ziele

            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }

#if defined(DEBUG) || defined(_DEBUG)
        if (Settings::debugLogging) {
            std::printf("[DEBUG] Best tile score: %.8f\n", bestScore);
        }
#endif

        if (bestIndex >= 0) {
            int bx = bestIndex % tilesX;
            int by = bestIndex / tilesX;

            float centerX = (bx + 0.5f) * tileSize;
            float centerY = (by + 0.5f) * tileSize;

            float2 tileCenter = {
                (centerX - width  / 2.0f) / zoom + offset.x,
                (centerY - height / 2.0f) / zoom + offset.y
            };

            newOffset = tileCenter;
            shouldZoom = true; // Struktur bleibt erhalten

        } else {
            shouldZoom = false;
        }
    } else {
        shouldZoom = false;
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
