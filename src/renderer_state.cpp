// Datei: src/renderer_state.cpp
// Zeilen: 101
// 🐭 Maus-Kommentar: Zustand des Renderers – jetzt mit geglättetem Ziel per EMA. `filteredTargetOffset` puffert sanft. Schneefuchs: „Ein Otter schlägt nicht abrupt den Kurs – er lässt Strömung zu.“
// Patch Schneefuchs Punkt 3: `cudaFree` wird jetzt sauber mit `CUDA_CHECK` abgesichert.
// 🐼 Panda integriert: setupCudaBuffers und resize verwalten jetzt auch d_contrast / h_contrast

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
    offset = { static_cast<double>(Settings::initialOffsetX), static_cast<double>(Settings::initialOffsetY) };

    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations = Settings::MAX_ITERATIONS_CAP;

    targetOffset = make_double2(offset.x, offset.y);
    filteredTargetOffset = { offset.x, offset.y };

    currentFPS = 0.0f;
    deltaTime = 0.0f;
    lastTileSize = Settings::BASE_TILE_SIZE;

    frameCount = 0;
    lastTime = glfwGetTime();

    supersampling = Settings::defaultSupersampling;
    overlayEnabled = false;
    lastTileIndex = -1;
}

void RendererState::updateOffsetTarget(double2 newOffset) {
    constexpr double alpha = 0.2;

    filteredTargetOffset.x = (1.0 - alpha) * filteredTargetOffset.x + alpha * static_cast<double>(newOffset.x);
    filteredTargetOffset.y = (1.0 - alpha) * filteredTargetOffset.y + alpha * static_cast<double>(newOffset.y);

    targetOffset = make_double2(filteredTargetOffset.x, filteredTargetOffset.y);
}

void RendererState::adaptIterationCount() {
    double logZoom = std::log10(zoom);
    maxIterations = static_cast<int>(baseIterations + logZoom * 200.0);
    maxIterations = std::min(maxIterations, Settings::MAX_ITERATIONS_CAP);
}

void RendererState::setupCudaBuffers() {
    const int totalPixels = width * height;
    const int tileSize = lastTileSize;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    CUDA_CHECK(cudaMalloc(&d_iterations, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_entropy,    numTiles   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_contrast,   numTiles   * sizeof(float)));  // 🐼 Panda: GPU-Kontrastdaten

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);  // 🐼 Panda: Host-Kontrastpuffer
}

void RendererState::resize(int newWidth, int newHeight) {
    if (d_iterations) {
        CUDA_CHECK(cudaFree(d_iterations));
        d_iterations = nullptr;
    }
    if (d_entropy) {
        CUDA_CHECK(cudaFree(d_entropy));
        d_entropy = nullptr;
    }
    if (d_contrast) {
        CUDA_CHECK(cudaFree(d_contrast));  // 🐼 Panda: Kontrast freigeben
        d_contrast = nullptr;
    }

    CudaInterop::unregisterPBO();

    if (pbo != 0) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (tex != 0) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }

    width = newWidth;
    height = newHeight;

    pbo = OpenGLUtils::createPBO(width, height);
    tex = OpenGLUtils::createTexture(width, height);

    CudaInterop::registerPBO(pbo);

    setupCudaBuffers();

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Resize auf %dx%d abgeschlossen\n", width, height);
    }
}
