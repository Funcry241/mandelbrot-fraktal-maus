// Datei: src/cuda_interop.cu

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "settings.hpp"
#include "core_kernel.h"
#include "memory_utils.hpp"
#include "progressive.hpp"

namespace CudaInterop {

#define CHECK_CUDA_STEP(call, msg) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err)); \
    } \
}

#define DEBUG_PRINT(fmt, ...) \
    do { if (Settings::debugLogging) { std::fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__); } } while(0)

void renderCudaFrame(
    cudaGraphicsResource_t cudaPboRes,
    int                   width,
    int                   height,
    float&                zoom,
    float2&               offset,
    int                   maxIter,
    float*                d_complexity,
    std::vector<float>&   h_complexity
) {
    DEBUG_PRINT("Starte Frame-Render");

    uchar4* d_img = nullptr;
    size_t  imgSize = 0;

    CHECK_CUDA_STEP(cudaGraphicsMapResources(1, &cudaPboRes), "cudaGraphicsMapResources");
    CHECK_CUDA_STEP(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_img), &imgSize, cudaPboRes), "cudaGraphicsResourceGetMappedPointer");

    if (Settings::debugGradient) {
        DEBUG_PRINT("Starte Debug-Gradient");
        launch_debugGradient(d_img, width, height);
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "DebugGradient Synchronize");
    } else {
        DEBUG_PRINT("Starte Mandelbrot-Kernel");
        launch_mandelbrotHybrid(d_img, width, height, zoom, offset, maxIter);
        CHECK_CUDA_STEP(cudaGetLastError(), "launch_mandelbrotHybrid");

        int totalTiles = static_cast<int>(h_complexity.size());

        CHECK_CUDA_STEP(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "cudaMemset d_complexity");

        dim3 blockDim(Settings::TILE_W, Settings::TILE_H);
        dim3 gridDim((width + Settings::TILE_W - 1) / Settings::TILE_W,
                     (height + Settings::TILE_H - 1) / Settings::TILE_H);

        DEBUG_PRINT("Starte Complexity-Kernel mit Grid (%d,%d) Block (%d,%d)", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        CHECK_CUDA_STEP(cudaGetLastError(), "computeComplexity Kernel-Start");
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "computeComplexity Synchronize");

        CHECK_CUDA_STEP(cudaMemcpy(h_complexity.data(), d_complexity, totalTiles * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_complexity->h_complexity");

        DEBUG_PRINT("Suche Bereich mit größter Varianz...");

        int tilesX = (width + Settings::TILE_W - 1) / Settings::TILE_W;
        float bestVariance = -1.0f;
        int   bestIdx = 0;

        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestVariance) {
                bestVariance = h_complexity[i];
                bestIdx = i;
            }
        }

        DEBUG_PRINT("Beste gefundene Varianz: %.6f", bestVariance);

        bool offsetChanged = false;
        bool zoomChanged = false;

        if (bestVariance > 0.0f) {
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            float newOffX = offset.x + ((bx + 0.5f) * Settings::TILE_W - width * 0.5f) / zoom;
            float newOffY = offset.y + ((by + 0.5f) * Settings::TILE_H - height * 0.5f) / zoom;

            offsetChanged = (std::fabs(newOffX - offset.x) > 1e-6f) || (std::fabs(newOffY - offset.y) > 1e-6f);
            if (std::isfinite(newOffX) && std::isfinite(newOffY)) {
                offset.x = newOffX;
                offset.y = newOffY;
            }

            DEBUG_PRINT("Neue Offset-Position: (%.6f, %.6f)", offset.x, offset.y);
        }

        float newZoom = zoom * Settings::zoomFactor;
        constexpr float maxZoomAllowed = 1e15f; // 🐭 deutlich höher für Deep-Zooms

        zoomChanged = (std::fabs(newZoom - zoom) > 1e-6f);
        if (std::isfinite(newZoom) && newZoom < maxZoomAllowed) {
            zoom = newZoom;
            DEBUG_PRINT("Neuer Zoom: %.6f", zoom);
        }

        if (offsetChanged || zoomChanged) {
            DEBUG_PRINT("Zoom oder Offset geändert — Iterationen werden zurückgesetzt.");
            resetIterations();
        }
    }

    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "cudaGraphicsUnmapResources");

    DEBUG_PRINT("Frame-Render abgeschlossen");
}

} // namespace CudaInterop
