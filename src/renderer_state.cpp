// Datei: src/renderer_state.cpp
// Zeilen: 98
// 🐭 Maus-Kommentar: Kolibri integriert! Buffer-Init jetzt immer robust. Keine Schattenwerte, keine toten Felder (lastIndex entfernt). Schneefuchs: „Kein Schattenwert bleibt im System.“ Otter validiert für Capybara v2.

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

    targetOffset   = make_double2(offset.x, offset.y);
    filteredTargetOffset = { offset.x, offset.y };

    currentFPS = 0.0f;
    deltaTime  = 0.0f;
    lastTileSize = Settings::BASE_TILE_SIZE;

    frameCount = 0;
    lastTime   = glfwGetTime();

    supersampling  = Settings::defaultSupersampling;
    overlayEnabled = false;
    lastTileIndex  = -1;
}

void RendererState::setupCudaBuffers() {
    const int totalPixels = width * height;
    const int tileSize = lastTileSize;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    CUDA_CHECK(cudaMalloc(&d_iterations, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int))); // Immer 0 – keine Schattenwerte!

    CUDA_CHECK(cudaMalloc(&d_entropy,    numTiles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_contrast,   numTiles * sizeof(float)));     // Panda
    CUDA_CHECK(cudaMalloc(&d_tileSupersampling, numTiles * sizeof(int))); // Kolibri

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);          // Panda
    h_tileSupersampling.resize(numTiles);  // Kolibri
}

void RendererState::resize(int newWidth, int newHeight) {
    if (d_iterations) { CUDA_CHECK(cudaFree(d_iterations)); d_iterations = nullptr; }
    if (d_entropy) { CUDA_CHECK(cudaFree(d_entropy)); d_entropy = nullptr; }
    if (d_contrast) { CUDA_CHECK(cudaFree(d_contrast)); d_contrast = nullptr; }
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
        CUDA_CHECK(cudaMemset(d_iterations, 0, width * height * sizeof(int))); // Nach Resize immer nullen!

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Resize auf %dx%d abgeschlossen\n", width, height);
    }
}
