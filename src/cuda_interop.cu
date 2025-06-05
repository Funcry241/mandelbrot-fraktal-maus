// Datei: src/cuda_interop.cu

// 🐭 Maus-Kommentar: Umgebaut auf RUNTIME-Wechsel per Settings::debugGradient, plus Reset bei Zoom-/Offset-Change

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

// Debug-Utilities
#define CHECK_CUDA_STEP(call, msg) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err)); \
    } \
}

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
    std::fprintf(stdout, "[INFO] Starte Frame-Render\n");

    // 1) PBO mappen → d_img holen
    uchar4* d_img = nullptr;
    size_t  imgSize = 0;

    CHECK_CUDA_STEP(cudaGraphicsMapResources(1, &cudaPboRes), "cudaGraphicsMapResources");
    CHECK_CUDA_STEP(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_img), &imgSize, cudaPboRes), "cudaGraphicsResourceGetMappedPointer");

    if (Settings::debugGradient) {
        std::fprintf(stdout, "[INFO] Starte Debug-Gradient\n");
        launch_debugGradient(d_img, width, height);
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "DebugGradient Synchronize");
    } else {
        std::fprintf(stdout, "[INFO] Starte Mandelbrot-Kernel\n");
        launch_mandelbrotHybrid(d_img, width, height, zoom, offset, maxIter);
        CHECK_CUDA_STEP(cudaGetLastError(), "launch_mandelbrotHybrid");

        int totalTiles = static_cast<int>(h_complexity.size());

        CHECK_CUDA_STEP(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "cudaMemset d_complexity");

        dim3 blockDim(Settings::TILE_W, Settings::TILE_H);
        dim3 gridDim((width + Settings::TILE_W - 1) / Settings::TILE_W,
                     (height + Settings::TILE_H - 1) / Settings::TILE_H);
        std::fprintf(stdout, "[INFO] Starte Complexity-Kernel mit Grid (%d,%d) Block (%d,%d)\n",
            gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        CHECK_CUDA_STEP(cudaGetLastError(), "computeComplexity Kernel-Start");
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "computeComplexity Synchronize");

        CHECK_CUDA_STEP(cudaMemcpy(h_complexity.data(), d_complexity, totalTiles * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_complexity->h_complexity");

        std::fprintf(stdout, "[INFO] Suche Bereich mit höchster Komplexität...\n");

        int tilesX = (width + Settings::TILE_W - 1) / Settings::TILE_W;
        float bestScore = -1.0f;
        int   bestIdx   = 0;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx   = i;
            }
        }

        bool offsetChanged = false;
        bool zoomChanged = false;

        if (bestScore > 0.0f) {
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            float newOffX = offset.x + ((bx + 0.5f) * Settings::TILE_W - width * 0.5f) / zoom;
            float newOffY = offset.y + ((by + 0.5f) * Settings::TILE_H - height * 0.5f) / zoom;
            offsetChanged = (std::fabs(newOffX - offset.x) > 1e-6f) || (std::fabs(newOffY - offset.y) > 1e-6f);
            if (std::isfinite(newOffX) && std::isfinite(newOffY)) {
                offset.x = newOffX;
                offset.y = newOffY;
            }
            std::fprintf(stdout, "[INFO] Neue Offset-Position: (%.6f, %.6f)\n", offset.x, offset.y);
        }

        float newZoom = zoom * Settings::zoomFactor;
        constexpr float maxZoomAllowed = 1e6f;
        zoomChanged = (std::fabs(newZoom - zoom) > 1e-6f);
        if (std::isfinite(newZoom) && newZoom < maxZoomAllowed) {
            zoom = newZoom;
            std::fprintf(stdout, "[INFO] Neuer Zoom: %.6f\n", zoom);
        }

        // Reset Iterationen, wenn Zoom oder Offset sich geändert haben
        if (offsetChanged || zoomChanged) {
            std::fprintf(stdout, "[INFO] Zoom oder Offset geändert — Iterationen werden zurückgesetzt.\n");
            resetIterations();
        }
    }

    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "cudaGraphicsUnmapResources");

    std::fprintf(stdout, "[INFO] Frame-Render abgeschlossen\n");
}

} // namespace CudaInterop
