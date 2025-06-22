// 🐭 Maus-Kommentar: Zustand des Renderers – jetzt mit korrekt initialisierter Zeitbasis. Keine Delta-Geister mehr beim ersten Frame. Schneefuchs: „Wer bei null beginnt, hat schon verloren.“

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"  // 🧱 Für PBO/Texture-Helfer

RendererState::RendererState(int w, int h)
    : width(w), height(h) {
    reset();
}

void RendererState::reset() {
    zoom = Settings::initialZoom;
    offset = { Settings::initialOffsetX, Settings::initialOffsetY };

    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations = Settings::MAX_ITERATIONS_CAP;

    targetOffset = offset;

    currentFPS = 0.0f;
    deltaTime = 0.0f;
    lastTileSize = Settings::BASE_TILE_SIZE;

    frameCount = 0;
    lastTime = static_cast<float>(glfwGetTime());  // 🟢 Statt 0.0 – echte Zeitbasis!
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

    h_entropy.resize(numTiles);
}

void RendererState::resize(int newWidth, int newHeight) {
    // 🧼 Alte CUDA-Puffer freigeben
    if (d_iterations) {
        cudaFree(d_iterations);
        d_iterations = nullptr;
    }
    if (d_entropy) {
        cudaFree(d_entropy);
        d_entropy = nullptr;
    }

    // 🧽 PBO deregistrieren
    CudaInterop::unregisterPBO();

    // 🗑️ Alte OpenGL-Ressourcen löschen
    if (pbo != 0) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (tex != 0) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }

    // 📐 Neue Größe setzen
    width = newWidth;
    height = newHeight;

    // 🆕 Neue Ressourcen erzeugen
    pbo = OpenGLUtils::createPBO(width, height);
    tex = OpenGLUtils::createTexture(width, height);

    // 🔗 CUDA-Interop neu registrieren
    CudaInterop::registerPBO(pbo);

    // 🔁 CUDA-Puffer neu allokieren
    setupCudaBuffers();

    // 🔒 TileSize stabilisieren – verhindert Resize-Loop
    lastTileSize = computeTileSizeFromZoom(zoom);

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Resize auf %dx%d abgeschlossen\n", width, height);
    }
}
