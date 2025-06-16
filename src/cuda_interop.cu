// Datei: src/cuda_interop.cu
// Zeilen: 123
// 🐭 Maus-Kommentar: CUDA/OpenGL-Interop für PBO-Mapping & Fraktalberechnung. Auto-Zoom via Entropieanalyse (pro Tile), bestes Tile wird ermittelt, Zoom-Ziel sanft interpoliert (LERP), Zoom nur bei Mindestdistanz. Pause-Funktion steuerbar via Tastendruck (P/Leertaste). Schneefuchs hätte auf `std::sqrt()` bestanden.

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
            if (h_entropy[i] > bestScore) {
                bestScore = h_entropy[i];
                bestIndex = i;
            }
        }

#if defined(DEBUG) || defined(_DEBUG)
        if (Settings::debugLogging) {
            std::printf("[DEBUG] Best tile entropy: %.8f\n", bestScore);
        }
#endif

        if (bestIndex >= 0) {
            int bx = bestIndex % tilesX;
            int by = bestIndex / tilesX;

            float scaleX = 1.0f / (zoom * width);
            float scaleY = 1.0f / (zoom * height);
            float2 tileCenter = {
                offset.x + (bx + 0.5f) * tileSize * scaleX,
                offset.y + (by + 0.5f) * tileSize * scaleY
            };

            float2 delta = {
                tileCenter.x - offset.x,
                tileCenter.y - offset.y
            };

            float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

            if (dist > Settings::MIN_JUMP_DISTANCE) {
                newOffset.x = offset.x + delta.x * Settings::LERP_FACTOR;
                newOffset.y = offset.y + delta.y * Settings::LERP_FACTOR;
                shouldZoom = true;
            } else {
                shouldZoom = false;
            }
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
