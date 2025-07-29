// 🐭 Maus-Kommentar: Keine Doppelregistrierung mehr - resize() übernimmt Verantwortung klar und kontrolliert.
// 🦦 Otter: Device-Buffers und PBO sauber, kein Zombie-Handle mehr.
// 🦊 Schneefuchs: Ressourcenfluss ist konsistent und deterministisch.

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

    // 🐭 Maus: Initialisierung exakt in Struct-Reihenfolge gemäß zoom_logic.hpp
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

void RendererState::setupCudaBuffers() {
    const int totalPixels = width * height;
    const int tileSize    = lastTileSize;
    const int tilesX      = (width + tileSize - 1) / tileSize;
    const int tilesY      = (height + tileSize - 1) / tileSize;
    const int numTiles    = tilesX * tilesY;

    CUDA_CHECK(cudaMalloc(&d_iterations, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_iterations, 0, totalPixels * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_entropy,  numTiles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_contrast, numTiles * sizeof(float)));

    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);
}

void RendererState::resize(int newWidth, int newHeight) {
    // Device-Ressourcen freigeben
    if (d_iterations) { CUDA_CHECK(cudaFree(d_iterations)); d_iterations = nullptr; }
    if (d_entropy)    { CUDA_CHECK(cudaFree(d_entropy));    d_entropy    = nullptr; }
    if (d_contrast)   { CUDA_CHECK(cudaFree(d_contrast));   d_contrast   = nullptr; }

    // CUDA-seitige Bindung zum alten PBO lösen
    CudaInterop::unregisterPBO();

    // OpenGL-Ressourcen löschen
    if (pbo) { glDeleteBuffers(1, &pbo); pbo = 0; }
    if (tex) { glDeleteTextures(1, &tex); tex = 0; }

    // Neue Größe setzen
    width  = newWidth;
    height = newHeight;

    // Neu erzeugen & registrieren
    OpenGLUtils::setGLResourceContext("resize");
    pbo = OpenGLUtils::createPBO(width, height);
    tex = OpenGLUtils::createTexture(width, height);
    CudaInterop::registerPBO(pbo);

    setupCudaBuffers();
    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Resize] %d x %d buffers reallocated", width, height);
}
