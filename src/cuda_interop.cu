// 🐭 CUDA-OpenGL Interop – brutal komprimiert für maximale Geschwindigkeit

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

#include "settings.hpp"
#include "core_kernel.h"
#include "memory_utils.hpp"
#include "progressive.hpp"

namespace CudaInterop {

#define CHECK_CUDA_STEP(call, msg) if (auto e = (call); e != cudaSuccess) throw std::runtime_error("[CUDA] " + std::string(msg) + ": " + cudaGetErrorString(e))
#define DEBUG_PRINT(fmt, ...) if (Settings::debugLogging) std::fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__)

void renderCudaFrame(cudaGraphicsResource_t cudaPboRes, int w, int h, float& zoom, float2& offset, int maxIter, float* d_comp, std::vector<float>& h_comp, int* d_iters) {
    DEBUG_PRINT("Starting frame render");
    uchar4* d_img = nullptr; size_t imgSize = 0;
    CHECK_CUDA_STEP(cudaGraphicsMapResources(1, &cudaPboRes), "MapResources");
    CHECK_CUDA_STEP(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &imgSize, cudaPboRes), "GetMappedPointer");

    if (Settings::debugGradient) {
        DEBUG_PRINT("Launching debug kernel");
        launch_debugGradient(d_img, w, h);
    } else {
        launch_mandelbrotHybrid(d_img, d_iters, w, h, zoom, offset, maxIter);
        int totalTiles = h_comp.size();
        CHECK_CUDA_STEP(cudaMemset(d_comp, 0, totalTiles * sizeof(float)), "Memset complexity");

        dim3 block(Settings::TILE_W, Settings::TILE_H), grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        DEBUG_PRINT("Launching complexity kernel Grid(%d, %d) Block(%d, %d)", grid.x, grid.y, block.x, block.y);

        float dynThreshold = Settings::dynamicVarianceThreshold(zoom);
        computeComplexity<<<grid, block>>>(d_iters, w, h, d_comp, dynThreshold);
        CHECK_CUDA_STEP(cudaDeviceSynchronize(), "Complexity sync");
        CHECK_CUDA_STEP(cudaMemcpy(h_comp.data(), d_comp, totalTiles * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy complexity");

        DEBUG_PRINT("Searching best tile...");
        int tilesX = (w + Settings::TILE_W - 1) / Settings::TILE_W, bestIdx = -1;
        float bestVar = -1.0f;

        for (int i = 0; i < totalTiles; ++i)
            if (h_comp[i] > dynThreshold && h_comp[i] > bestVar) bestVar = h_comp[i], bestIdx = i;

        if (bestIdx != -1) {
            DEBUG_PRINT("Best variance: %.12f", bestVar);
            int bx = bestIdx % tilesX, by = bestIdx / tilesX;
            float tx = (bx + 0.5f) * Settings::TILE_W - w * 0.5f, ty = (by + 0.5f) * Settings::TILE_H - h * 0.5f;
            float targetX = offset.x + tx / zoom, targetY = offset.y + ty / zoom;

            if (std::isfinite(targetX) && std::isfinite(targetY)) {
                auto step = [](float d, float f, float z) {
                    float s = fminf(fmaxf(fabsf(d), fmaxf(Settings::MIN_OFFSET_STEP, 1e-5f / z)), f / z);
                    return d > 0.0f ? s : -s;
                };
                offset.x += step(targetX - offset.x, Settings::OFFSET_STEP_FACTOR, zoom);
                offset.y += step(targetY - offset.y, Settings::OFFSET_STEP_FACTOR, zoom);
                DEBUG_PRINT("New offset: (%.12f, %.12f)", offset.x, offset.y);
            }

            float targetZoom = zoom * Settings::zoomFactor;
            if (std::isfinite(targetZoom) && targetZoom < 1e15f) {
                float delta = targetZoom - zoom;
                zoom += (delta > 0.0f ? 1 : -1) * fminf(fmaxf(fabsf(delta), Settings::MIN_ZOOM_STEP), Settings::ZOOM_STEP_FACTOR * zoom);
                DEBUG_PRINT("New zoom: %.12f", zoom);
            }
        }
    }
    CHECK_CUDA_STEP(cudaGraphicsUnmapResources(1, &cudaPboRes), "UnmapResources");
    DEBUG_PRINT("Frame render complete");
}

} // namespace CudaInterop
