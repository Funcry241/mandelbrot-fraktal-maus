// Datei: src/renderer_state.cpp
// 🐭 Maus-Kommentar: Tile-Größe jetzt sichtbar. Kein malloc ins Leere mehr.
// 🦦 Otter: Fehler sichtbar, deterministisch, kein division-by-zero.
// 🐜 Rote Ameise: setupCudaBuffers nimmt tileSize explizit entgegen – Datenfluss 100% klar.
// 🐑 Hirte: Validierung via cudaPointerGetAttributes – wenn’s kracht, wissen wir was d_entropy wirklich ist.
// 🦊 Schneefuchs: Wenn es kracht, wissen wir exakt wo.

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

    targetOffset         = offset;
    filteredTargetOffset = offset;

    fps        = 0.0f;
    deltaTime  = 0.0f;
    frameCount = 0;
    lastTime   = glfwGetTime();

    lastTileSize = Settings::BASE_TILE_SIZE;

    heatmapOverlayEnabled       = Settings::heatmapOverlayEnabled;
    warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;

    zoomResult.bestIndex       = -1;
    zoomResult.bestEntropy     = 0.0f;
    zoomResult.bestContrast    = 0.0f;
    zoomResult.bestScore       = 0.0f;
    zoomResult.distance        = 0.0f;
    zoomResult.minDistance     = 0.0f;
    zoomResult.relEntropyGain  = 0.0f;
    zoomResult.relContrastGain = 0.0f;
    zoomResult.isNewTarget     = false;
    zoomResult.shouldZoom      = false;
    zoomResult.newOffset       = offset;
    zoomResult.perTileContrast.clear();
}

void RendererState::setupCudaBuffers(int tileSize) {
    const int totalPixels = width * height;
    const int tilesX      = (width + tileSize - 1) / tileSize;
    const int tilesY      = (height + tileSize - 1) / tileSize;
    const int numTiles    = tilesX * tilesY;

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] setupCudaBuffers: %d x %d -> tileSize=%d -> %d tiles",
                       width, height, tileSize, numTiles);

    CUDA_CHECK(cudaSetDevice(0));
    CudaInterop::logCudaDeviceContext("setupCudaBuffers");

    // --- Iteration-Puffer ---
    CUDA_CHECK(cudaMalloc(&d_iterations, totalPixels * sizeof(int)));
    LUCHS_LOG_HOST("[CHECK] cudaMalloc d_iterations: err=%d -> %p (%d bytes)", 0, (void*)d_iterations, totalPixels * (int)sizeof(int));
    CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int)));
    LUCHS_LOG_HOST("[CHECK] cudaMemset d_iterations: err=%d", 0);

    // --- Entropy-Puffer ---
    CUDA_CHECK(cudaMalloc(&d_entropy, numTiles * sizeof(float)));
    LUCHS_LOG_HOST("[CHECK] cudaMalloc d_entropy: err=%d -> %p (%d bytes)", 0, (void*)d_entropy, numTiles * (int)sizeof(float));
    
    // 🐑 Hirte: Validierung der Device-Pointer-Eigenschaften nach malloc
    cudaPointerAttributes attr = {};
    cudaError_t attrErr = cudaPointerGetAttributes(&attr, d_entropy);
    LUCHS_LOG_HOST("[CHECK] d_entropy: attrErr=%d type=%d device=%d hostPtr=%p devicePtr=%p",
                   (int)attrErr, (int)attr.type, (int)attr.device,
                   (void*)attr.hostPointer, (void*)attr.devicePointer);

    CUDA_CHECK(cudaMemset(d_entropy, 0, numTiles * sizeof(float)));
    LUCHS_LOG_HOST("[CHECK] cudaMemset d_entropy: issued");

    // --- Synchronisation zur Absturzprüfung ---
    cudaDeviceSynchronize();
    cudaError_t syncErr = cudaGetLastError();
    LUCHS_LOG_HOST("[CHECK] cudaDeviceSynchronize after d_entropy memset: err=%d", (int)syncErr);
    if (syncErr != cudaSuccess)
        throw std::runtime_error("cudaMemset d_entropy failed (post-sync)");

    // --- Contrast-Puffer ---
    CUDA_CHECK(cudaMalloc(&d_contrast, numTiles * sizeof(float)));
    LUCHS_LOG_HOST("[CHECK] cudaMalloc d_contrast: err=%d -> %p (%d bytes)", 0, (void*)d_contrast, numTiles * (int)sizeof(float));
    CUDA_CHECK(cudaMemset(d_contrast, 0, numTiles * sizeof(float)));
    LUCHS_LOG_HOST("[CHECK] cudaMemset d_contrast: err=%d", 0);

    // --- Zusammenfassung ---
    LUCHS_LOG_HOST("[ALLOC] d_iterations=%p d_entropy=%p d_contrast=%p | %dx%d px -> tileSize=%d -> %d tiles",
                   (void*)d_iterations, (void*)d_entropy, (void*)d_contrast,
                   width, height, tileSize, numTiles);

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
}

void RendererState::resize(int newWidth, int newHeight) {
    if (d_iterations) { CUDA_CHECK(cudaFree(d_iterations)); d_iterations = nullptr; }
    if (d_entropy)    { CUDA_CHECK(cudaFree(d_entropy));    d_entropy    = nullptr; }
    if (d_contrast)   { CUDA_CHECK(cudaFree(d_contrast));   d_contrast   = nullptr; }

    CudaInterop::unregisterPBO();

    if (pbo) { glDeleteBuffers(1, &pbo); pbo = 0; }
    if (tex) { glDeleteTextures(1, &tex); tex = 0; }

    width  = newWidth;
    height = newHeight;

    OpenGLUtils::setGLResourceContext("resize");
    pbo = OpenGLUtils::createPBO(width, height);
    tex = OpenGLUtils::createTexture(width, height);
    CudaInterop::registerPBO(pbo);

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] resize(): zoom=%.5f → tileSize=%d", zoom, lastTileSize);

    setupCudaBuffers(lastTileSize);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Resize] %d x %d buffers reallocated", width, height);
}
