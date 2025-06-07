// Datei: src/cuda_interop.cu
// 🐭 Maus-Kommentar: Implementation für CUDA-OpenGL Interop mit Auto-Zoom Pan-Step Fix

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
#include <stdexcept>  // 🐭 Exception-basiertes Fehler-Handling

#include "settings.hpp"
#include "core_kernel.h"
#include "memory_utils.hpp"
#include "progressive.hpp"

namespace CudaInterop {

// 🚀 Fehlerbehandlung: Exception statt std::exit!
#define CHECK_CUDA_STEP(call, msg) do { \
    if (cudaError_t err = (call); err != cudaSuccess) { \
        throw std::runtime_error(std::string("[CUDA ERROR] ") + msg + ": " + cudaGetErrorString(err)); \
    } \
} while (0)

#define DEBUG_PRINT(fmt, ...) do { \
    if (Settings::debugLogging) \
        std::fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__); \
} while (0)

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

        if (bestIdx != -1) {
            DEBUG_PRINT("Best variance: %.12f", bestVariance);
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            float tx = (bx + 0.5f) * Settings::TILE_W - w * 0.5f;
            float ty = (by + 0.5f) * Settings::TILE_H - h * 0.5f;
            float targetOffX = offset.x + tx / zoom;
            float targetOffY = offset.y + ty / zoom;

            if (std::isfinite(targetOffX) && std::isfinite(targetOffY)) {
                auto step = [](float delta, float factor, float zoom) {
                    const float maxStep = factor / zoom;
                    const float minStep = fmaxf(Settings::MIN_OFFSET_STEP, 1e-5f / zoom);

                    float stepSize = fminf(fmaxf(std::fabs(delta), minStep), maxStep);  // 🐭 Clamping zwischen Min und Max
                    return (delta > 0.0f) ? stepSize : -stepSize;
                };

                offset.x += step(targetOffX - offset.x, Settings::OFFSET_STEP_FACTOR, zoom);
                offset.y += step(targetOffY - offset.y, Settings::OFFSET_STEP_FACTOR, zoom);
                DEBUG_PRINT("New offset: (%.12f, %.12f)", offset.x, offset.y);
            }

            float targetZoom = zoom * Settings::zoomFactor;
            if (std::isfinite(targetZoom) && targetZoom < 1e15f) {
                float zoomDelta = targetZoom - zoom;
                float maxZoomStep = Settings::ZOOM_STEP_FACTOR * zoom;
                float zoomStep = fminf(fmaxf(std::fabs(zoomDelta), Settings::MIN_ZOOM_STEP), maxZoomStep);  // 🐭 Clamping Zoom Step
                zoom += (zoomDelta > 0.0f) ? zoomStep : -zoomStep;
                DEBUG_PRINT("New zoom: %.12f", zoom);
            }
        }
    }
    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "UnmapResources");
    DEBUG_PRINT("Frame render complete");
}

// 🐭 Dynamic Parallelism Check entfernt — nicht mehr nötig!

} // namespace CudaInterop
