// Datei: src/renderer_state.cpp
// Zeilen: 56
// 🐭 Maus-Kommentar: Zustand des Renderers: Zoom, Offset, FPS, Iterationen – aber keine redundante Ressourceninitialisierung mehr. Schneefuchs: „State kümmert sich um Werte – nicht um Texturen!“

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"  // 💡 Enthält CUDA_CHECK

RendererState::RendererState(int w, int h)
    : width(w), height(h) {
    reset();
}

void RendererState::reset() {
    zoom = Settings::initialZoom;
    offset = { Settings::initialOffsetX, Settings::initialOffsetY };

    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations = Settings::MAX_ITERATIONS_CAP;

    targetZoom = zoom;
    targetOffset = offset;

    smoothedZoom = zoom;               // 🧈 verhindert Ruck nach Reset
    smoothedOffset = offset;

    currentFPS = 0.0f;
    deltaTime = 0.0f;
    lastTileSize = Settings::BASE_TILE_SIZE;

    // 🕒 Frame-Timing-Reset
    frameCount = 0;
    lastTime = 0.0;
    lastFrameTime = 0.0f;
}

void RendererState::updateZoomTarget(float newZoom) {
    targetZoom = newZoom;
}

void RendererState::updateOffsetTarget(float2 newOffset) {
    targetOffset = newOffset;
}

void RendererState::adaptIterationCount() {
    float logZoom = std::log10(zoom);
    maxIterations = static_cast<int>(baseIterations + logZoom * 200.0f);
    maxIterations = std::min(maxIterations, Settings::MAX_ITERATIONS_CAP);
}

void RendererState::setupCudaBuffers() {
    const int totalPixels = width * height;
    const int tileSize = lastTileSize;
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int numTiles = tilesX * tilesY;

    CUDA_CHECK(cudaMalloc(&d_iterations, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_entropy, numTiles * sizeof(float)));

    h_entropy.resize(numTiles);  // Optional: hostseitig vorallozieren
}
