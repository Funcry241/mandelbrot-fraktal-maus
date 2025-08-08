// Datei: src/renderer_state.cpp
// 🐭 Maus-Kommentar: Tile-Größe jetzt sichtbar. Kein malloc ins Leere mehr.
// 🦦 Otter: Fehler sichtbar, deterministisch, kein division-by-zero.
// 🐜 Schwarze Ameise: setupCudaBuffers nimmt tileSize explizit entgegen – Datenfluss 100% klar.
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
    zoom = static_cast<double>(Settings::initialZoom);
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
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] setupCudaBuffers params: width=%d height=%d zoom=%.5f tileSize=%d tilesX=%d tilesY=%d numTiles=%d totalPixels=%d",
                       width, height, zoom, tileSize, tilesX, tilesY, numTiles, totalPixels);
    }

    CUDA_CHECK(cudaSetDevice(0));
    CudaInterop::logCudaDeviceContext("setupCudaBuffers");

    // --- Iteration-Puffer ---
    d_iterations.allocate(totalPixels * sizeof(int));
    LUCHS_LOG_HOST("[CHECK] allocate d_iterations: ptr=%p size=%d bytes", d_iterations.get(), totalPixels * (int)sizeof(int));
    CUDA_CHECK(cudaMemset(d_iterations.get(), 0, totalPixels * sizeof(int)));
    LUCHS_LOG_HOST("[CHECK] cudaMemset d_iterations done");

    // --- Entropy-Puffer ---
    d_entropy.allocate(numTiles * sizeof(float));
    LUCHS_LOG_HOST("[CHECK] allocate d_entropy: ptr=%p size=%d bytes", d_entropy.get(), numTiles * (int)sizeof(float));

    cudaPointerAttributes attr = {};
    cudaError_t attrErr = cudaPointerGetAttributes(&attr, d_entropy.get());
    LUCHS_LOG_HOST("[CHECK] d_entropy: attrErr=%d type=%d device=%d hostPtr=%p devicePtr=%p",
                   (int)attrErr, (int)attr.type, (int)attr.device,
                   (void*)attr.hostPointer, (void*)attr.devicePointer);

    CUDA_CHECK(cudaMemset(d_entropy.get(), 0, numTiles * sizeof(float)));
    LUCHS_LOG_HOST("[CHECK] cudaMemset d_entropy done");

    cudaDeviceSynchronize();
    cudaError_t syncErr = cudaGetLastError();
    LUCHS_LOG_HOST("[CHECK] cudaDeviceSynchronize after d_entropy memset: err=%d", (int)syncErr);
    if (syncErr != cudaSuccess)
        throw std::runtime_error("cudaMemset d_entropy failed (post-sync)");

    // --- Contrast-Puffer ---
    d_contrast.allocate(numTiles * sizeof(float));
    LUCHS_LOG_HOST("[CHECK] allocate d_contrast: ptr=%p size=%d bytes", d_contrast.get(), numTiles * (int)sizeof(float));
    CUDA_CHECK(cudaMemset(d_contrast.get(), 0, numTiles * sizeof(float)));
    LUCHS_LOG_HOST("[CHECK] cudaMemset d_contrast done");

    // --- Zusammenfassung ---
    LUCHS_LOG_HOST("[ALLOC] d_iterations=%p d_entropy=%p d_contrast=%p | %dx%d px -> tileSize=%d -> %d tiles",
                   d_iterations.get(), d_entropy.get(), d_contrast.get(),
                   width, height, tileSize, numTiles);

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
}

void RendererState::resize(int newWidth, int newHeight) {
    // Alte CUDA-Device-Puffer freigeben
    d_iterations.free();
    d_entropy.free();
    d_contrast.free();

    // CUDA-Interop-PBO abmelden
    CudaInterop::unregisterPBO();

    // OpenGL-Puffer sicher freigeben via RAII
    pbo.free();
    tex.free();

    // Neue Größe setzen
    width  = newWidth;
    height = newHeight;

    OpenGLUtils::setGLResourceContext("resize");

    // Explizit GLBuffer-Wrapper aus GLuint erzeugen
    pbo = Hermelin::GLBuffer(OpenGLUtils::createPBO(width, height));
    tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(width, height));

    CudaInterop::registerPBO(pbo);

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[DEBUG] resize(): zoom=%.5f -> tileSize=%d", zoom, lastTileSize);

    setupCudaBuffers(lastTileSize);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Resize] %d x %d buffers reallocated", width, height);
}
