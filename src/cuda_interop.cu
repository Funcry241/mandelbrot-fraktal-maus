// 🐭 Maus-Kommentar: CUDA-OpenGL Interop mit sanftem Zoom- und Offset-Gliding inkl. Pause-Funktion mit Leertaste + Ziel-Glättung gegen Zittern

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "settings.hpp"
#include "core_kernel.h"
#include "memory_utils.hpp"
#include "progressive.hpp"

namespace CudaInterop {

#define CHECK_CUDA_STEP(call, msg) do { \
    if (cudaError_t err = (call); err != cudaSuccess) { \
        throw std::runtime_error(std::string("[CUDA ERROR] ") + msg + ": " + cudaGetErrorString(err)); \
    } \
} while (0)

#define DEBUG_PRINT(fmt, ...) do { \
    if (Settings::debugLogging) \
        std::fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__); \
} while (0)

static bool pauseZoom = false;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        pauseZoom = !pauseZoom;
        if (Settings::debugLogging)
            std::fprintf(stdout, "[DEBUG] Zoom %s\n", pauseZoom ? "paused" : "resumed");
    }
}

void renderCudaFrame(
    cudaGraphicsResource_t cudaPboRes,
    int w,
    int h,
    float& zoom,
    float2& offset,
    int maxIter,
    float* d_complexity,
    std::vector<float>& h_complexity,
    int* d_iterations,
    bool autoZoomEnabled
) {
    DEBUG_PRINT("Starting frame render");

    static float2 targetOffset = offset; // 🐭 Sanftes Ziel
    static float lastBestVariance = -1.0f; // 🐭 Letzte beste Varianz

    uchar4* d_img = nullptr;
    size_t imgSize = 0;
    CHECK_CUDA_STEP(cudaGraphicsMapResources(1, &cudaPboRes), "MapResources");
    CHECK_CUDA_STEP(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_img), &imgSize, cudaPboRes), "GetMappedPointer");

    if (Settings::debugGradient) {
        DEBUG_PRINT("Launching debug kernel");
        launch_debugGradient(d_img, w, h);
    } else {
        DEBUG_PRINT("Launching Mandelbrot kernel");
        launch_mandelbrotHybrid(d_img, d_iterations, w, h, zoom, offset, maxIter);

        int totalTiles = static_cast<int>(h_complexity.size());
        CHECK_CUDA_STEP(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "Memset complexity");

        dim3 blockDim(Settings::TILE_W, Settings::TILE_H);
        dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
        DEBUG_PRINT("Launching complexity kernel Grid(%d, %d) Block(%d, %d)", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        computeComplexity<<<gridDim, blockDim>>>(d_iterations, w, h, d_complexity);
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "complexity sync");
        CHECK_CUDA_STEP(cudaMemcpy(h_complexity.data(), d_complexity, totalTiles * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy complexity");

        int nonzeroTiles = 0;
        float maxComplexity = -1.0f;
        float minComplexity = 1e30f;
        float sumComplexity = 0.0f;

        for (int i = 0; i < totalTiles; ++i) {
            float val = h_complexity[i];
            if (val > 0.0f) {
                nonzeroTiles++;
                if (val > maxComplexity) maxComplexity = val;
                if (val < minComplexity) minComplexity = val;
                sumComplexity += val;
            }
        }

        float avgComplexity = (nonzeroTiles > 0) ? (sumComplexity / nonzeroTiles) : 0.0f;

        DEBUG_PRINT("Complexity Stats: Nonzero Tiles: %d / %d | Max: %.6e | Min: %.6e | Avg: %.6e", nonzeroTiles, totalTiles, maxComplexity, minComplexity, avgComplexity);

        DEBUG_PRINT("Searching best tile...");
        int tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W;
        float bestVariance = -1.0f;
        int bestIdx = -1;

        float dynamicThreshold = Settings::dynamicVarianceThreshold(zoom);

        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > dynamicThreshold && h_complexity[i] > bestVariance) {
                bestVariance = h_complexity[i];
                bestIdx = i;
            }
        }

        if (bestIdx == -1) {
            DEBUG_PRINT("No suitable tile found in current frame.");
        } else {
            DEBUG_PRINT("Best Tile Index: %d | Variance Score: %.6e", bestIdx, bestVariance);

            if (bestVariance > lastBestVariance * 1.02f || lastBestVariance < 0.0f) {
                lastBestVariance = bestVariance;

                int bx = bestIdx % tilesX;
                int by = bestIdx / tilesX;
                float tx = (bx + 0.5f) * Settings::TILE_W - w * 0.5f;
                float ty = (by + 0.5f) * Settings::TILE_H - h * 0.5f;

                float newTargetX = offset.x + tx / zoom;
                float newTargetY = offset.y + ty / zoom;

                if (std::isfinite(newTargetX) && std::isfinite(newTargetY)) {
                    targetOffset.x = newTargetX;
                    targetOffset.y = newTargetY;
                    DEBUG_PRINT("New target offset set: (%.12f, %.12f)", targetOffset.x, targetOffset.y);
                }
            }
        }

        float lerpFactor = 0.05f;
        offset.x += (targetOffset.x - offset.x) * lerpFactor;
        offset.y += (targetOffset.y - offset.y) * lerpFactor;
        DEBUG_PRINT("Smoothed offset: (%.12f, %.12f)", offset.x, offset.y);
    }

    if (autoZoomEnabled && !pauseZoom) {
        if (std::isfinite(zoom) && zoom < 1e15f) {
            zoom += Settings::ZOOM_STEP_FACTOR * zoom;
            DEBUG_PRINT("Zoom updated: %.12f", zoom);
        }
    }

    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "UnmapResources");
    DEBUG_PRINT("Frame render complete");
}

} // namespace CudaInterop
