// 🐭 Maus-Kommentar: CUDA-OpenGL Interop mit sanftem Auto-Zoom und Fallback bei "keinem besten Tile"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
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

// 🐭 Hilfsfunktion für sanftes Gleiten (linear interpolation)
inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

void renderCudaFrame(cudaGraphicsResource_t cudaPboRes, int w, int h, float& zoom, float2& offset,
                     int maxIter, float* d_complexity, std::vector<float>& h_complexity, int* d_iterations) {
    DEBUG_PRINT("Starting frame render");

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

        if (Settings::debugLogging) {
            int nonzeroTiles = 0;
            float maxComplexity = -1.0f;
            float minComplexity = 1e30f;
            float sumComplexity = 0.0f;

            for (int i = 0; i < totalTiles; ++i) {
                float val = h_complexity[i];
                if (val > 0.0f) {
                    nonzeroTiles++;
                    maxComplexity = fmaxf(maxComplexity, val);
                    minComplexity = fminf(minComplexity, val);
                    sumComplexity += val;
                }
            }

            float avgComplexity = (nonzeroTiles > 0) ? (sumComplexity / nonzeroTiles) : 0.0f;

            DEBUG_PRINT("Complexity Stats: Nonzero Tiles: %d / %d | Max: %.6e | Min: %.6e | Avg: %.6e",
                        nonzeroTiles, totalTiles, maxComplexity, minComplexity, avgComplexity);
        }

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

        float2 targetOffset = offset;
        float targetZoom = zoom;

        if (Settings::debugLogging) {
            if (bestIdx == -1) {
                DEBUG_PRINT("No suitable tile found in current frame.");
            } else {
                DEBUG_PRINT("Best Tile Index: %d | Variance Score: %.6e", bestIdx, bestVariance);
            }
        }

        if (bestIdx != -1) {
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            float tx = (bx + 0.5f) * Settings::TILE_W - w * 0.5f;
            float ty = (by + 0.5f) * Settings::TILE_H - h * 0.5f;
            targetOffset.x = offset.x + tx / zoom;
            targetOffset.y = offset.y + ty / zoom;

            if (!std::isfinite(targetOffset.x) || !std::isfinite(targetOffset.y)) {
                targetOffset = offset;  // 🐭 Fallback auf alten Wert
            }

            float newZoom = zoom * Settings::zoomFactor;
            if (std::isfinite(newZoom) && newZoom < 1e15f) {
                targetZoom = newZoom;
            }
        } else {
            // 🐭 Kein bestes Tile gefunden → trotzdem weiter reinzoomen
            targetZoom = zoom * 1.01f;  // 1% reinzoomen, langsam
        }

        // 🐭 Weiches Gleiten: 20% pro Frame Richtung Ziel
        constexpr float LERP_FACTOR = 0.2f;
        offset.x = lerp(offset.x, targetOffset.x, LERP_FACTOR);
        offset.y = lerp(offset.y, targetOffset.y, LERP_FACTOR);
        zoom     = lerp(zoom,     targetZoom,     LERP_FACTOR);

        DEBUG_PRINT("New offset: (%.12f, %.12f)", offset.x, offset.y);
        DEBUG_PRINT("New zoom: %.12f", zoom);
    }
    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "UnmapResources");
    DEBUG_PRINT("Frame render complete");
}

} // namespace CudaInterop
