// Datei: src/renderer_state.cpp
// 🐭 Maus-Kommentar: Alpha 49d – Flugente watschelt voran: `targetOffset` ist jetzt `float2`, alles konsistent zur GPU. Kein `double2` mehr, kein Konvertierungskrach. Resize-Log bleibt kompakt. Otter lacht.

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"

RendererState::RendererState(int w, int h)
: width(w), height(h) {
    reset();
}

void RendererState::reset() {
    zoom   = static_cast<double>(Settings::initialZoom);
    offset = { static_cast<float>(Settings::initialOffsetX), static_cast<float>(Settings::initialOffsetY) };

    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations  = Settings::MAX_ITERATIONS_CAP;

    targetOffset         = offset;  // 🦆 Flugente: float2 statt double2
    filteredTargetOffset = offset;

    currentFPS   = 0.0f;
    deltaTime    = 0.0f;
    frameCount   = 0;
    lastTime     = glfwGetTime();
    lastTileSize = Settings::BASE_TILE_SIZE;

    supersampling  = Settings::defaultSupersampling;
    overlayEnabled = Settings::heatmapOverlayEnabled;

    // 🔄 ZoomResult vollständig zurücksetzen (Gepard)
    zoomResult.bestIndex    = -1;
    zoomResult.bestEntropy  = 0.0f;
    zoomResult.bestContrast = 0.0f;
    zoomResult.newOffset    = offset;
    zoomResult.shouldZoom   = false;
    zoomResult.isNewTarget  = false;
}

void RendererState::setupCudaBuffers() {
    const int totalPixels = width * height;
    const int tileSize    = lastTileSize;
    const int tilesX      = (width + tileSize - 1) / tileSize;
    const int tilesY      = (height + tileSize - 1) / tileSize;
    const int numTiles    = tilesX * tilesY;

    CUDA_CHECK(cudaMalloc(&d_iterations, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_entropy,             numTiles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_contrast,            numTiles * sizeof(float)));       // Panda
    CUDA_CHECK(cudaMalloc(&d_tileSupersampling,   numTiles * sizeof(int)));         // Kolibri

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
    h_tileSupersampling.resize(numTiles);
}

void RendererState::resize(int newWidth, int newHeight) {
    if (d_iterations)        { CUDA_CHECK(cudaFree(d_iterations));        d_iterations = nullptr; }
    if (d_entropy)           { CUDA_CHECK(cudaFree(d_entropy));           d_entropy = nullptr; }
    if (d_contrast)          { CUDA_CHECK(cudaFree(d_contrast));          d_contrast = nullptr; }
    if (d_tileSupersampling) { CUDA_CHECK(cudaFree(d_tileSupersampling)); d_tileSupersampling = nullptr; }

    CudaInterop::unregisterPBO();

    if (pbo != 0) { glDeleteBuffers(1, &pbo); pbo = 0; }
    if (tex != 0) { glDeleteTextures(1, &tex); tex = 0; }

    width  = newWidth;
    height = newHeight;

    pbo = OpenGLUtils::createPBO(width, height);
    tex = OpenGLUtils::createTexture(width, height);

    CudaInterop::registerPBO(pbo);
    setupCudaBuffers();

    if (d_iterations)
        CUDA_CHECK(cudaMemset(d_iterations, 0, width * height * sizeof(int)));

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));

    if (Settings::debugLogging)
        std::printf("[Resize] %d x %d buffers reallocated\n", width, height);
}
