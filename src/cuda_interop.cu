// 🍝 Maus-Kommentar: CUDA-Interop für Mandelbrot-Renderer –
// verwaltet PBO-Mapping, Fraktal-Rendering und adaptive Auto-Zoom-Logik.

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

// 🌐 Interne CUDA-Resource für das PBO-Mapping
static cudaGraphicsResource_t cudaResource;

// 💤 Zustand: Ist Auto-Zoom pausiert?
static bool pauseZoom = false;

// 🔌 PBO bei CUDA registrieren
void registerPBO(GLuint pbo) {
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

// 🧹 PBO von CUDA deregistrieren
void unregisterPBO() {
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
}

// 🖼️ Hauptfunktion: CUDA-Fraktal-Frame rendern und analysieren
void renderCudaFrame(uchar4* pbo, int* d_iterations, float* d_stddev, float* d_mean,
                     int width, int height, float zoom, float2 offset, int maxIterations,
                     std::vector<float>& h_complexity, float2& outNewOffset, bool& shouldZoom,
                     int tileSize) {
    // 🔁 CUDA <-> OpenGL: PBO mappen
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));
    uchar4* devPtr;
    size_t size;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource));

    // 🔨 1. Fraktal berechnen
    launch_mandelbrotHybrid(devPtr, d_iterations, width, height, zoom, offset, maxIterations);

    // 📊 2. Komplexitätsanalyse starten
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int totalTiles = tilesX * tilesY;
    computeComplexity(d_iterations, d_mean, d_stddev, width, height, tileSize);
    CUDA_CHECK(cudaMemcpy(h_complexity.data(), d_stddev, totalTiles * sizeof(float), cudaMemcpyDeviceToHost));

    // 🔎 3. Beste Kachel für Auto-Zoom bestimmen
    float bestScore = -1.0f;
    float2 bestTileOffset = {0.0f, 0.0f};
    shouldZoom = false;

    for (int tileY = 0; tileY < tilesY; ++tileY) {
        for (int tileX = 0; tileX < tilesX; ++tileX) {
            int tileIndex = tileY * tilesX + tileX;
            float gradient = h_complexity[tileIndex];

            if (gradient < Settings::dynamicVarianceThreshold(zoom)) continue;

            // 🧭 Position der Kachel-Mitte in Pixel
            float pixelX = (tileX + 0.5f) * tileSize;
            float pixelY = (tileY + 0.5f) * tileSize;

            // 🔄 Umrechnen auf Fraktalkoordinaten
            float2 tileOffset = {
                offset.x + (pixelX - width * 0.5f) / zoom,
                offset.y + (pixelY - height * 0.5f) / zoom
            };

            // 📐 Entfernung zur aktuellen Ansicht
            float tileDist = std::hypot(tileOffset.x - offset.x, tileOffset.y - offset.y);

            // 🧠 Bonus: Nähe zum Ursprung der Mandelbrotmenge
            float distToCenter = std::hypot(tileOffset.x + 0.75f, tileOffset.y);
            float centralityBoost = 1.0f / (distToCenter + 0.1f);

            // 📈 Score für diese Kachel
            float score = gradient * centralityBoost / (tileDist + 1.0f);

            if (score > bestScore) {
                bestScore = score;
                bestTileOffset = tileOffset;
                shouldZoom = true;
            }
        }
    }

    if (shouldZoom) {
        outNewOffset = bestTileOffset;
    }

    // 🔁 CUDA <-> OpenGL: PBO unmap
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
}

// 🧭 Getter: Ist Auto-Zoom pausiert?
bool getPauseZoom() {
    return pauseZoom;
}

// 🧭 Setter: Auto-Zoom pausieren oder fortsetzen
void setPauseZoom(bool paused) {
    pauseZoom = paused;
}

}  // namespace CudaInterop
