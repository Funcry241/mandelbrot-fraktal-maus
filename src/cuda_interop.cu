// Datei: src/cuda_interop.cu
// 🍝 Maus-Kommentar: Auto-Zoom mit Gradient-Erkennung für wirklich interessante Fraktalbereiche

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
#include <algorithm>

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
static bool autoZoomEnabled = true;

void setPauseZoom(bool state) { pauseZoom = state; }
bool getPauseZoom() { return pauseZoom; }
bool getAutoZoomEnabled() { return autoZoomEnabled; }

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_SPACE:
                autoZoomEnabled = !autoZoomEnabled;
                std::printf("[INFO] Auto-Zoom %s\n", autoZoomEnabled ? "ENABLED" : "DISABLED");
                break;
            case GLFW_KEY_P:
                pauseZoom = !pauseZoom;
                std::printf("[INFO] Zoom %s\n", pauseZoom ? "PAUSED" : "RESUMED");
                break;
            default:
                break;
        }
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
    bool autoZoomEnabledParam
) {
    DEBUG_PRINT("Starting frame render");

    static float2 targetOffset = offset;
    static float lastBestGradient = -1.0f;

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

        int tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W;
        int tilesY = (h + Settings::TILE_H - 1) / Settings::TILE_H;
        int currTileX = static_cast<int>((offset.x * zoom + w * 0.5f) / Settings::TILE_W);
        int currTileY = static_cast<int>((offset.y * zoom + h * 0.5f) / Settings::TILE_H);

        int dynamicRadius = static_cast<int>(std::sqrt(zoom) * Settings::DYNAMIC_RADIUS_SCALE);
        dynamicRadius = std::clamp(dynamicRadius, Settings::DYNAMIC_RADIUS_MIN, Settings::DYNAMIC_RADIUS_MAX);

        DEBUG_PRINT("Search Radius: %d", dynamicRadius);

        float bestGradient = -1.0f;
        int bestIdx = -1;

        for (int dy = -dynamicRadius; dy <= dynamicRadius; ++dy) {
            for (int dx = -dynamicRadius; dx <= dynamicRadius; ++dx) {
                if (dx * dx + dy * dy > dynamicRadius * dynamicRadius) continue;
                int tx = currTileX + dx;
                int ty = currTileY + dy;
                if (tx >= 0 && ty >= 0 && tx < tilesX && ty < tilesY) {
                    int idx = ty * tilesX + tx;
                    float variance = h_complexity[idx];

                    float neighborVariance = 0.0f;
                    int neighborCount = 0;
                    for (int ny = -1; ny <= 1; ++ny) {
                        for (int nx = -1; nx <= 1; ++nx) {
                            if (nx == 0 && ny == 0) continue;
                            int ntx = tx + nx;
                            int nty = ty + ny;
                            if (ntx >= 0 && nty >= 0 && ntx < tilesX && nty < tilesY) {
                                int nidx = nty * tilesX + ntx;
                                neighborVariance += h_complexity[nidx];
                                neighborCount++;
                            }
                        }
                    }
                    if (neighborCount > 0) neighborVariance /= neighborCount;

                    float gradient = fabsf(variance - neighborVariance);
                    float dist2 = dx * dx + dy * dy + 1e-5f;
                    float score = gradient / dist2;

                    if (score > bestGradient) {
                        bestGradient = score;
                        bestIdx = idx;
                    }
                }
            }
        }

        if (bestIdx != -1 && bestGradient > lastBestGradient * 1.02f) {
            lastBestGradient = bestGradient;
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            float tx = (bx + 0.5f) * Settings::TILE_W - w * 0.5f;
            float ty = (by + 0.5f) * Settings::TILE_H - h * 0.5f;
            float newTargetX = offset.x + tx / zoom;
            float newTargetY = offset.y + ty / zoom;
            if (std::isfinite(newTargetX) && std::isfinite(newTargetY)) {
                targetOffset = { newTargetX, newTargetY };
                DEBUG_PRINT("New Target Offset: (%.12f, %.12f)", targetOffset.x, targetOffset.y);
            }
        } else {
            DEBUG_PRINT("No better tile found — continuing.");
        }

        offset.x += (targetOffset.x - offset.x) * Settings::LERP_FACTOR;
        offset.y += (targetOffset.y - offset.y) * Settings::LERP_FACTOR;
        DEBUG_PRINT("Smoothed Offset: (%.12f, %.12f)", offset.x, offset.y);
    }

    if (autoZoomEnabledParam && !pauseZoom) {
        if (std::isfinite(zoom) && zoom < 1e18f) {
            zoom += Settings::ZOOM_STEP_FACTOR * zoom;
            DEBUG_PRINT("Zoom Updated: %.12f", zoom);
        }
    }

    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "UnmapResources");
    DEBUG_PRINT("Frame render complete");
}

} // namespace CudaInterop
