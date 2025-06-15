// ASCII-Only CUDA-Interop für Mandelbrot-Renderer – PBO-Mapping, Fraktal-Rendering & Auto-Zoom mit Entropieanalyse

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
#include <algorithm>

#include "settings.hpp"
#include "core_kernel.h"
#include "memory_utils.hpp"
#include "progressive.hpp"
#include "common.hpp"

namespace CudaInterop {

static cudaGraphicsResource_t cudaResource = nullptr;
static bool pauseZoom = false;

// ✂️ Deregistriert PBO von CUDA – notwendig bei Resize oder Shutdown
void unregisterPBO() {
    if (cudaResource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
        cudaResource = nullptr;
    }
}

// 🔗 Registriert neues OpenGL-PBO bei CUDA
void registerPBO(GLuint pbo) {
    if (cudaResource) unregisterPBO();
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

// 🚀 Hauptfunktion für CUDA-Frame-Rendering inkl. Entropieanalyse pro Tile
void renderCudaFrame(uchar4*, int* d_iterations, float* d_entropy, float* d_stddev,
                     int width, int height, float zoom, float2 offset, int maxIter,
                     std::vector<float>& h_entropy, float2& newOffset, bool& shouldZoom, int tileSize) {

    if (!cudaResource) {
        std::fprintf(stderr, "[ERROR] CUDA resource not registered!\n");
        return;
    }

    // 🔄 CUDA<->OpenGL Mapping
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));
    uchar4* devPtr;
    size_t size;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource));

    // 🌀 CUDA-Kernel starten (Fraktal + Entropie)
    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIter);
    computeTileEntropy(d_iterations, d_entropy, width, height, tileSize, maxIter);

    // 📊 Host-seitige Entropie-Puffer vorbereiten
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int totalTiles = tilesX * tilesY;

    h_entropy.resize(totalTiles);
    CUDA_CHECK(cudaMemcpy(h_entropy.data(), d_entropy, totalTiles * sizeof(float), cudaMemcpyDeviceToHost));

    // 📉 Entropie-Diagnose
#if defined(DEBUG) || Settings::debugLogging
    float minE = 1e10f, maxE = -1.0f, sumE = 0.0f;
    for (int i = 0; i < totalTiles; ++i) {
        float e = h_entropy[i];
        minE = std::min(minE, e);
        maxE = std::max(maxE, e);
        sumE += e;
    }
    float meanE = sumE / totalTiles;
    float threshold = Settings::dynamicVarianceThreshold(zoom);
    std::printf("[DEBUG] Entropy stats: min=%.12f | max=%.12f | mean=%.12f | threshold=%.12f\n",
                minE, maxE, meanE, threshold);
#else
    float threshold = Settings::dynamicVarianceThreshold(zoom);
#endif

    // 🔍 Beste Zoom-Region bestimmen
    float bestScore = -1.0f;
    float2 bestOffset = {};
    shouldZoom = false;

    for (int y = 0; y < tilesY; ++y) {
        for (int x = 0; x < tilesX; ++x) {
            int idx = y * tilesX + x;
            float entropy = h_entropy[idx];
            if (entropy < threshold) continue;

            float2 cand = {
                offset.x + ((x + 0.5f) * tileSize - width * 0.5f) / zoom,
                offset.y + ((y + 0.5f) * tileSize - height * 0.5f) / zoom
            };

            float dist = std::hypot(cand.x - offset.x, cand.y - offset.y);
            float cent = std::hypot(cand.x + 0.75f, cand.y);
            float score = entropy / (dist + 1.0f) / (cent + 0.1f);

            if (score > bestScore) {
                bestScore = score;
                bestOffset = cand;
                shouldZoom = true;
            }
        }
    }

    // 🧭 Neue Zielkoordinaten setzen (falls sinnvoll)
    if (shouldZoom) {
#if defined(DEBUG)
        std::printf("[ZOOM] Best score = %.10f (threshold = %.10f)\n", bestScore, threshold);
#endif
        newOffset = bestOffset;
    }

    // 🔄 CUDA<->OpenGL Unmapping
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
}

// 🛑 Zoom-Pause-Toggle via HUD/Keybinding
bool getPauseZoom() { return pauseZoom; }
void setPauseZoom(bool p) { pauseZoom = p; }

// ⌨️ SPACE oder P zum Pausieren der Auto-Zoom-Logik
void keyCallback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_SPACE || key == GLFW_KEY_P) {
        pauseZoom = !pauseZoom;
#if defined(DEBUG)
        std::printf("[INFO] Auto-Zoom %s\n", pauseZoom ? "PAUSIERT" : "AKTIV");
#endif
    }
}

} // namespace CudaInterop
