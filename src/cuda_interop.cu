// Datei: src/cuda_interop.cu

#pragma once

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

#include "settings.hpp"
#include "core_kernel.h"
#include "memory_utils.hpp"
#include "progressive.hpp"

namespace CudaInterop {

#define CHECK_CUDA_STEP(call, msg) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
}

#define DEBUG_PRINT(fmt, ...) \
    do { if (Settings::debugLogging) { std::fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__); } } while(0)

void renderCudaFrame(
    cudaGraphicsResource_t cudaPboRes,
    int width,
    int height,
    float& zoom,
    float2& offset,
    int maxIter,
    float* d_complexity,
    std::vector<float>& h_complexity
) {
    DEBUG_PRINT("Starting frame render");

    uchar4* d_img = nullptr;
    size_t  imgSize = 0;

    CHECK_CUDA_STEP(cudaGraphicsMapResources(1, &cudaPboRes), "cudaGraphicsMapResources");
    CHECK_CUDA_STEP(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_img), &imgSize, cudaPboRes), "cudaGraphicsResourceGetMappedPointer");

    if (Settings::debugGradient) {
        DEBUG_PRINT("Launching debug gradient kernel");
        launch_debugGradient(d_img, width, height);
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "DebugGradient synchronize");
    } else {
        DEBUG_PRINT("Launching Mandelbrot kernel");
        launch_mandelbrotHybrid(d_img, width, height, zoom, offset, maxIter);
        CHECK_CUDA_STEP(cudaGetLastError(), "launch_mandelbrotHybrid");

        int totalTiles = static_cast<int>(h_complexity.size());

        CHECK_CUDA_STEP(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "cudaMemset d_complexity");

        dim3 blockDim(Settings::TILE_W, Settings::TILE_H);
        dim3 gridDim((width + Settings::TILE_W - 1) / Settings::TILE_W,
                     (height + Settings::TILE_H - 1) / Settings::TILE_H);

        DEBUG_PRINT("Launching complexity kernel with Grid (%d, %d) Block (%d, %d)", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        CHECK_CUDA_STEP(cudaGetLastError(), "computeComplexity kernel launch");
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "computeComplexity synchronize");

        CHECK_CUDA_STEP(cudaMemcpy(h_complexity.data(), d_complexity, totalTiles * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_complexity -> h_complexity");

        DEBUG_PRINT("Searching for tile with highest variance...");

        int tilesX = (width + Settings::TILE_W - 1) / Settings::TILE_W;
        float bestVariance = -1.0f;
        int   bestIdx = 0;

        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestVariance) {
                bestVariance = h_complexity[i];
                bestIdx = i;
            }
        }

        DEBUG_PRINT("Best variance found: %.6f", bestVariance);

        if (bestVariance > 0.0f) {
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            float targetOffX = offset.x + ((bx + 0.5f) * Settings::TILE_W - width * 0.5f) / zoom;
            float targetOffY = offset.y + ((by + 0.5f) * Settings::TILE_H - height * 0.5f) / zoom;

            if (std::isfinite(targetOffX) && std::isfinite(targetOffY)) {
                // Maus-Sanftheit: Limitiere Offset-Delta
                float deltaX = targetOffX - offset.x;
                float deltaY = targetOffY - offset.y;
                const float maxOffsetStep = Settings::OFFSET_STEP_FACTOR / zoom;

                if (std::fabs(deltaX) > maxOffsetStep) deltaX = (deltaX > 0.0f ? maxOffsetStep : -maxOffsetStep);
                if (std::fabs(deltaY) > maxOffsetStep) deltaY = (deltaY > 0.0f ? maxOffsetStep : -maxOffsetStep);

                offset.x += deltaX;
                offset.y += deltaY;
                DEBUG_PRINT("New offset: (%.6f, %.6f)", offset.x, offset.y);
            }

            float targetZoom = zoom * Settings::zoomFactor;
            constexpr float maxZoomAllowed = 1e15f;

            if (std::isfinite(targetZoom) && targetZoom < maxZoomAllowed) {
                // Maus-Sanftheit: Limitiere Zoom-Delta
                float zoomDelta = targetZoom - zoom;
                const float maxZoomStep  = Settings::ZOOM_STEP_FACTOR * zoom;

                if (std::fabs(zoomDelta) > maxZoomStep) {
                    zoomDelta = (zoomDelta > 0.0f ? maxZoomStep : -maxZoomStep);
                }

                zoom += zoomDelta;
                DEBUG_PRINT("New zoom: %.6f", zoom);
            }
        }
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "[SYNC ERROR] Before unmap: %s\n", cudaGetErrorString(syncErr));
        std::exit(EXIT_FAILURE);
    }

    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "cudaGraphicsUnmapResources");

    DEBUG_PRINT("Frame render complete");
}

void checkDynamicParallelismSupport() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        std::fprintf(stderr, "Dynamic Parallelism not supported. Compute Capability 3.5+ required.\n");
        std::exit(EXIT_FAILURE);
    }
}

} // namespace CudaInterop
