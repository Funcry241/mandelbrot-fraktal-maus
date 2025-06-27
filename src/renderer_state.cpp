// Datei: src/renderer_state.cpp
// Zeilen: 93
// 🐭 Maus-Kommentar: Zustand des Renderers – jetzt mit geglättetem Ziel per EMA. `filteredTargetOffset` puffert sanft. Schneefuchs: „Ein Otter schlägt nicht abrupt den Kurs – er lässt Strömung zu.“
// Patch Schneefuchs Punkt 3: `cudaFree` wird jetzt sauber mit `CUDA_CHECK` abgesichert.

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
    zoom = static_cast<double>(Settings::initialZoom);
    offset = { static_cast<double>(Settings::initialOffsetX), static_cast<double>(Settings::initialOffsetY) };

    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations = Settings::MAX_ITERATIONS_CAP;

    targetOffset = make_double2(static_cast<float>(offset.x), static_cast<float>(offset.y));
    filteredTargetOffset = { offset.x, offset.y };  // 🆕 EMA-Initialisierung

    currentFPS = 0.0f;
    deltaTime = 0.0f;
    lastTileSize = Settings::BASE_TILE_SIZE;

    frameCount = 0;
    lastTime = glfwGetTime();  // 🔄 Präzise als double speichern
}

void RendererState::updateOffsetTarget(double2 newOffset) {
    constexpr double alpha = 0.2;  // 🧮 Glättungsfaktor: kleiner = langsamer, weicher

    // 💧 Exponentieller Filter auf double-Basis
    filteredTargetOffset.x = (1.0 - alpha) * filteredTargetOffset.x + alpha * static_cast<double>(newOffset.x);
    filteredTargetOffset.y = (1.0 - alpha) * filteredTargetOffset.y + alpha * static_cast<double>(newOffset.y);

    // ⛵ Zielposition für Kamera: weich verfolgt
    targetOffset = make_double2(
        static_cast<float>(filteredTargetOffset.x),
        static_cast<float>(filteredTargetOffset.y)
    );
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
    CUDA_CHECK(cudaMalloc(&d_entropy, numTiles * sizeof(float)));

    h_entropy.resize(numTiles);
}

void RendererState::resize(int newWidth, int newHeight) {
    // 🧼 Alte CUDA-Puffer freigeben
    if (d_iterations) {
        CUDA_CHECK(cudaFree(d_iterations));  // ✅ Sicher freigeben
        d_iterations = nullptr;
    }
    if (d_entropy) {
        CUDA_CHECK(cudaFree(d_entropy));     // ✅ Sicher freigeben
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
    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Resize auf %dx%d abgeschlossen\n", width, height);
    }
}
