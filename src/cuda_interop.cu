#include "pch.hpp"  // 💡 Muss als erstes stehen!
#include "cuda_interop.hpp"
#include "core_kernel.h"
#include "settings.hpp"
#include "common.hpp"
#include "renderer_state.hpp"

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

float computeEntropyContrast(const std::vector<float>& h, int index, int tilesX, int tilesY) {
    int x = index % tilesX;
    int y = index / tilesX;
    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < tilesX && ny >= 0 && ny < tilesY) {
                int nIndex = ny * tilesX + nx;
                sum += fabsf(h[index] - h[nIndex]);
                ++count;
            }
        }
    }
    return count > 0 ? sum / count : 0.0f;
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
        const float dynamicThreshold = std::max(Settings::VARIANCE_THRESHOLD / std::log2(zoom_f + 2.0f), Settings::MIN_VARIANCE_THRESHOLD);

        float2 bestOffset = offset_f;
        float bestScore = -1.0f;
        float bestEntropy = 0.0f;
        float bestContrast = 0.0f;
        int bestIndex = -1;

        constexpr float alpha = 1.0f;
        constexpr float beta = 1.5f;

        for (int i = 0; i < numTiles; ++i) {
            float e = h_entropy[i];
            if (e <= dynamicThreshold) continue;

            float c = computeEntropyContrast(h_entropy, i, tilesX, tilesY);
            float score = alpha * e + beta * c;

            if (score > bestScore) {
                int bx = i % tilesX;
                int by = i / tilesX;

                float centerX = (bx + 0.5f) * tileSize;
                float centerY = (by + 0.5f) * tileSize;

                bestOffset = {
                    (centerX - width / 2.0f) / zoom_f + offset_f.x,
                    (centerY - height / 2.0f) / zoom_f + offset_f.y
                };

                bestScore = score;
                bestEntropy = e;
                bestContrast = c;
                bestIndex = i;
            }
        }

        float2 prevTarget = state.smoothedTargetOffset;
        float2 delta = {
            bestOffset.x - prevTarget.x,
            bestOffset.y - prevTarget.y
        };
        float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

        constexpr float MIN_PIXEL_JUMP = 1.0f;
        float minDist = MIN_PIXEL_JUMP / zoom_f;
        bool isNewTarget = bestIndex >= 0 && dist > minDist;

        if (isNewTarget) {
            state.smoothedTargetOffset = bestOffset;
            state.smoothedTargetScore = bestScore;
        }

        if (bestIndex >= 0) {
            newOffset = state.smoothedTargetOffset;
            shouldZoom = true;
        }

#if ENABLE_ZOOM_LOGGING
        std::printf(
            "ZoomLog Z %.5e Th %.6f Idx %d Ent %.5f C %.5f S %.5f Dist %.6f Min %.6f New %d\n",
            zoom_f, dynamicThreshold, bestIndex, bestEntropy, bestContrast,
            bestScore, dist, minDist, isNewTarget ? 1 : 0
        );
#endif
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
