// Datei: src/renderer_loop.cpp
// Zeilen: 335
// 🐭 Maus-Kommentar: Komplett gefixt: Alle potenziellen double→float-Konvertierungen explizit gecastet.
// Otter sagt: "No more hidden warnings, nur klare Tatsachen."
// Clean/Rebuild empfohlen, um alte PCH-Kollisionen zu vermeiden.

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"
#include "settings.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "heatmap_overlay.hpp"
#include "frame_pipeline.hpp"
#include "zoom_command.hpp"

namespace RendererLoop {

static FrameContext ctx;
static CommandBus zoomBus;
static bool isFirstFrame = true;

void initResources(RendererState& state) {
if (state.pbo != 0 || state.tex != 0) {
if (Settings::debugLogging) {
std::puts("[DEBUG] initResources() skipped - resources already initialized");
}
return;
}

OpenGLUtils::setGLResourceContext("init");
state.pbo = OpenGLUtils::createPBO(state.width, state.height);
state.tex = OpenGLUtils::createTexture(state.width, state.height);

CudaInterop::registerPBO(state.pbo);
Hud::init();

state.lastTileSize = computeTileSizeFromZoom(static_cast<float>(state.zoom));
state.setupCudaBuffers();

if (Settings::debugLogging) {
    std::puts("[DEBUG] initResources() completed");
}

}

void beginFrame(RendererState& state) {
// Fix C4244: alle double->float Casts explizit
float currentTime = static_cast<float>(glfwGetTime());
float lastTimeFloat = static_cast<float>(state.lastTime);

float delta = currentTime - lastTimeFloat;
if (delta < 0.0f) delta = 0.0f;  // Sicherung gegen Zeitrücksprung

state.deltaTime = delta;
state.lastTime = static_cast<double>(currentTime);  // state.lastTime bleibt double

state.frameCount++;
if (state.deltaTime > 0.0f) {
    state.currentFPS = 1.0f / state.deltaTime;
}

}

// OtterFix: autoZoomEnabled und andere unbenutzte Parameter als unused markieren
void renderFrame_impl(RendererState& state, bool autoZoomEnabled) {
(void)autoZoomEnabled; // unused

if (isFirstFrame) {
    ctx.zoom = static_cast<float>(state.zoom);
    ctx.offset.x = static_cast<float>(state.offset.x);
    ctx.offset.y = static_cast<float>(state.offset.y);
    isFirstFrame = false;
}

// Context aktualisieren mit expliziten Casts bei Bedarf
ctx.width = state.width;
ctx.height = state.height;
ctx.maxIterations = state.maxIterations;
ctx.tileSize = state.lastTileSize;
ctx.supersampling = state.supersampling;
ctx.d_iterations = state.d_iterations;
ctx.d_entropy = state.d_entropy;
ctx.d_contrast = state.d_contrast;
ctx.h_entropy = state.h_entropy;
ctx.h_contrast = state.h_contrast;
ctx.overlayActive = state.overlayEnabled;
ctx.lastEntropy = state.lastEntropy;
ctx.lastContrast = state.lastContrast;
ctx.lastTileIndex = state.lastTileIndex;

beginFrame(state);

// --- Maus: Vor jedem CUDA-Frame Iterationsbuffer, Entropy und Contrast Nullen!
size_t totalPixels = static_cast<size_t>(ctx.width) * static_cast<size_t>(ctx.height);
size_t tilesX = (ctx.width + ctx.tileSize - 1) / ctx.tileSize;
size_t tilesY = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
size_t tilesCount = tilesX * tilesY;
CUDA_CHECK(cudaMemset(ctx.d_iterations, 0, totalPixels * sizeof(int)));
CUDA_CHECK(cudaMemset(ctx.d_entropy, 0, tilesCount * sizeof(float)));
CUDA_CHECK(cudaMemset(ctx.d_contrast, 0, tilesCount * sizeof(float)));

// 1. CUDA Fraktalberechnung & Iterations-Buffer aktualisieren
computeCudaFrame(ctx, state);

// 2. Fraktalbild anzeigen (Bilddaten aus PBO → Texture)
RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height);
drawFrame(ctx, state.tex, state);

// 3. Heatmap/Overlay-Analyse erst jetzt! (Kiwi)
{
    // Hier nochmal Nullen für Heatmap-Kohärenz
    CUDA_CHECK(cudaMemset(ctx.d_entropy, 0, tilesCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx.d_contrast, 0, tilesCount * sizeof(float)));

    CudaInterop::computeCudaEntropyContrast(
        ctx.d_iterations,
        ctx.d_entropy,
        ctx.d_contrast,
        ctx.width,
        ctx.height,
        ctx.tileSize,
        ctx.maxIterations
    );

    CUDA_CHECK(cudaMemcpy(ctx.h_entropy.data(), ctx.d_entropy, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_contrast.data(), ctx.d_contrast, tilesCount * sizeof(float), cudaMemcpyDeviceToHost));
}

if (Settings::debugLogging) {
    float e0 = ctx.h_entropy.empty() ? 0.0f : ctx.h_entropy[0];
    float c0 = ctx.h_contrast.empty() ? 0.0f : ctx.h_contrast[0];
    std::printf("[Heatmap] Entropy[0]=%.4f Contrast[0]=%.4f\n", e0, c0);
}

// 4. Overlay-Rendering (jetzt wirklich synchron zur aktuellen Iteration!)
HeatmapOverlay::drawOverlay(
    ctx.h_entropy,
    ctx.h_contrast,
    ctx.width,
    ctx.height,
    ctx.tileSize,
    0,       // textureId unused
    state    // aktueller RendererState
);

Hud::draw(state);

// State zurückschreiben mit explizitem Cast
state.zoom = static_cast<double>(ctx.zoom);
state.offset.x = static_cast<float>(ctx.offset.x);
state.offset.y = static_cast<float>(ctx.offset.y);
state.h_entropy = ctx.h_entropy;
state.h_contrast = ctx.h_contrast;
state.shouldZoom = ctx.shouldZoom;
state.lastEntropy = ctx.lastEntropy;
state.lastContrast = ctx.lastContrast;
state.lastTileIndex = ctx.lastTileIndex;

}

// OtterFix: unbenutzte Parameter als unused markieren
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
(void)scancode; (void)mods;

if (action != GLFW_PRESS) return;

RendererState* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
if (!state) return;

switch (key) {
    case GLFW_KEY_H:
        HeatmapOverlay::toggle(*state);
        break;
    case GLFW_KEY_P:
        CudaInterop::setPauseZoom(!CudaInterop::getPauseZoom());
        break;
    default:
        break;
}

}

} // namespace RendererLoop
