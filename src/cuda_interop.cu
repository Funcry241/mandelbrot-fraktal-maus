// Datei: src/cuda_interop.cu
// Zeilen: 207
// 🐅 Maus-Kommentar: CUDA/OpenGL-Interop für PBO-Mapping & Fraktalberechnung. Jetzt mit geglätteter Zielverfolgung (`smoothedTargetOffset`) – kein Zucken mehr! Schneefuchs: „Ein guter Otter verfolgt nur ein Ziel – aber mit Anstand.“

#include "pch.hpp"  // 💡 Muss als erstes stehen!
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"  // 🧠 Zugriff auf smoothedTargetOffset

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
    float zoom,
    float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize
) {
    if (!cudaPboResource) {
        std::cerr << "[FATAL] CUDA PBO not registered before renderCudaFrame.\n";
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    uchar4* devPtr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations);
    computeTileEntropy(d_iterations, d_entropy, width, height, tileSize, maxIterations);

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    h_entropy.resize(numTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

    shouldZoom = false;

    if (!pauseZoom) {
        const float dynamicThreshold = std::max(Settings::VARIANCE_THRESHOLD / std::log2(zoom + 2.0f), Settings::MIN_VARIANCE_THRESHOLD);

        float2 bestOffset = offset;
        float bestEntropy = 0.0f;
        float bestScore = -1.0f;
        int bestIndex = -1;

        for (int i = 0; i < numTiles; ++i) {
            int bx = i % tilesX;
            int by = i / tilesX;

            float centerX = (bx + 0.5f) * tileSize;
            float centerY = (by + 0.5f) * tileSize;

            float2 tileCenter = {
                (centerX - width  / 2.0f) / zoom + offset.x,
                (centerY - height / 2.0f) / zoom + offset.y
            };

            float2 delta = { tileCenter.x - offset.x, tileCenter.y - offset.y };
            float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
            float score = h_entropy[i] / (1.0f + Settings::ENTROPY_NEARBY_BIAS * dist);

            if (h_entropy[i] > dynamicThreshold && score > bestScore) {
                bestScore = score;
                bestOffset = tileCenter;
                bestEntropy = h_entropy[i];
                bestIndex = i;
            }
        }

        static int lastIndex = -1;
        if (Settings::debugLogging && bestIndex != lastIndex) {
            std::printf("[DEBUG] Zoom = %.6f | Threshold = %.8f\n", zoom, dynamicThreshold);
            if (bestIndex >= 0) {
                std::printf("[DEBUG] Best Tile = %d | Score = %.6f | Entropy = %.6f\n", bestIndex, bestScore, bestEntropy);
            } else {
                std::puts("[DEBUG] No tile passed threshold. Zoom paused.");
            }
            lastIndex = bestIndex;
        }

        if (bestIndex >= 0) {
            // Zugriff auf RendererState
            extern RendererState* globalRendererState;
            auto& state = *globalRendererState;

            constexpr float SMOOTHING_ALPHA   = 0.15f;
            constexpr float SCORE_THRESHOLD   = 0.95f;
            constexpr float NEWTARGET_DIST    = 0.001f;

            float2 delta = {
                bestOffset.x - state.smoothedTargetOffset.x,
                bestOffset.y - state.smoothedTargetOffset.y
            };
            float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
            bool isNewTarget = bestScore > state.smoothedTargetScore * SCORE_THRESHOLD || dist > NEWTARGET_DIST;

            if (isNewTarget) {
                state.smoothedTargetOffset = bestOffset;
                state.smoothedTargetScore = bestScore;
                state.framesSinceTargetChange = 0;
            } else {
                state.framesSinceTargetChange++;
            }

            newOffset = state.smoothedTargetOffset;
            shouldZoom = true;
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
    if (action == GLFW_PRESS && (key == GLFW_KEY_P || key == GLFW_KEY_SPACE)) {
        pauseZoom = !pauseZoom;
        std::cout << "[Zoom] Auto-Zoom " << (pauseZoom ? "paused" : "resumed") << "\n";
    }
}

} // namespace CudaInterop
