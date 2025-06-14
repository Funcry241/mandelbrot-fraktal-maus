// 🍝 Maus-Kommentar: CUDA-Interop für Mandelbrot-Renderer –
// verwaltet PBO-Mapping, Fraktal-Rendering, adaptive Komplexitätsbewertung & Auto-Zoom-Logik.

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
#include "common.hpp"

namespace CudaInterop {

static cudaGraphicsResource_t cudaResource = nullptr;  // 🔗 CUDA ↔ OpenGL Interop-Handle
static bool pauseZoom = false;                         // ⏸️ Auto-Zoom pausiert?

// 🧼 Deregistriert PBO aus CUDA
void unregisterPBO() {
    if (cudaResource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
        cudaResource = nullptr;
    }
}

// 🧼 Registriert OpenGL-PBO für CUDA-Zugriff
void registerPBO(GLuint pbo) {
    if (cudaResource) {
        unregisterPBO();
    }
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

// 🖼️ Rendert CUDA-Fraktal-Frame & analysiert Komplexität (Auto-Zoom)
void renderCudaFrame(uchar4* pbo,
                     int* d_iterations,
                     float* d_complexity,
                     float* d_stddev,
                     int width,
                     int height,
                     float zoom,
                     float2 offset,
                     int maxIterations,
                     const std::vector<float>& h_complexity,
                     float2& outNewOffset,
                     bool& shouldZoom,
                     int tileSize)
{
    if (!cudaResource) {
        std::fprintf(stderr, "[ERROR] CUDA resource not registered!\n");
        return;
    }

    if (Settings::debugLogging) {
        std::printf("[DEBUG] cuda_interop: renderCudaFrame\n");
        std::printf("         zoom: %.10f\n", zoom);
        std::printf("         offset: (%.10f, %.10f)\n", offset.x, offset.y);
        std::printf("         iterations: %d\n", maxIterations);
        std::printf("         tileSize: %d\n", tileSize);
        std::printf("         image: %d x %d\n", width, height);
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));
    if (Settings::debugLogging) std::puts("[DEBUG] cuda_interop: PBO mapped.");

    uchar4* devPtr;
    size_t size;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource));

    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations);
    if (Settings::debugLogging) std::puts("[DEBUG] cuda_interop: Mandelbrot kernel launched.");

    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int totalTiles = tilesX * tilesY;

    computeTileEntropy(d_iterations, d_stddev, width, height, tileSize, maxIterations);
    if (Settings::debugLogging) std::puts("[DEBUG] cuda_interop: Entropy kernel launched.");

    if (h_complexity.size() != static_cast<size_t>(totalTiles)) {
        std::fprintf(stderr, "[WARN] h_complexity resize: %zu → %d\n", h_complexity.size(), totalTiles);
        const_cast<std::vector<float>&>(h_complexity).resize(totalTiles);
    }

    CUDA_CHECK(cudaMemcpy((void*)h_complexity.data(), d_stddev, totalTiles * sizeof(float), cudaMemcpyDeviceToHost));
    if (Settings::debugLogging) std::puts("[DEBUG] cuda_interop: Entropy copied to host.");

    float bestScore = -1.0f;
    float2 bestTileOffset = {0.0f, 0.0f};
    shouldZoom = false;

    for (int tileY = 0; tileY < tilesY; ++tileY) {
        for (int tileX = 0; tileX < tilesX; ++tileX) {
            int tileIndex = tileY * tilesX + tileX;
            float entropy = h_complexity[tileIndex];

            if (entropy < Settings::dynamicVarianceThreshold(zoom)) continue;

            float pixelX = (tileX + 0.5f) * tileSize;
            float pixelY = (tileY + 0.5f) * tileSize;

            float2 tileOffset = {
                offset.x + (pixelX - width * 0.5f) / zoom,
                offset.y + (pixelY - height * 0.5f) / zoom
            };

            float tileDist = std::hypot(tileOffset.x - offset.x, tileOffset.y - offset.y);
            float distToCenter = std::hypot(tileOffset.x + 0.75f, tileOffset.y);
            float centralityBoost = 1.0f / (distToCenter + 0.1f);
            float score = entropy * centralityBoost / (tileDist + 1.0f);

            if (score > bestScore) {
                bestScore = score;
                bestTileOffset = tileOffset;
                shouldZoom = true;
            }
        }
    }

    if (shouldZoom) {
        outNewOffset = bestTileOffset;
        if (Settings::debugLogging) {
            std::puts("[DEBUG] Zoom decision: YES");
            std::printf("         └─ newOffset: (%.10f, %.10f)\n", outNewOffset.x, outNewOffset.y);
        }
    } else {
        if (Settings::debugLogging) {
            std::puts("[DEBUG] Zoom decision: NO");
        }
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
    if (Settings::debugLogging) std::puts("[DEBUG] cuda_interop: PBO unmapped.");
}

bool getPauseZoom() {
    return pauseZoom;
}

void setPauseZoom(bool paused) {
    pauseZoom = paused;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    switch (key) {
        case GLFW_KEY_SPACE:
        case GLFW_KEY_P:
            pauseZoom = !pauseZoom;
#if defined(DEBUG) || Settings::debugLogging
            std::printf("[INFO] Taste %s gedrückt – Auto-Zoom %s\n",
                        key == GLFW_KEY_SPACE ? "SPACE" : "P",
                        pauseZoom ? "PAUSIERT" : "AKTIV");
#endif
            break;
        default:
            break;
    }
}

}  // namespace CudaInterop
