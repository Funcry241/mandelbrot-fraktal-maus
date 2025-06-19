// Datei: src/renderer_core.cu
// Zeilen: 84
// 🐭 Maus-Kommentar: Entry-Point fürs Rendering. Jetzt mit `glewInit()` direkt nach Kontext-Erstellung und automatischem Cleanup im Destruktor. Schneefuchs: „Nur wer gründlich aufräumt, darf Neues entstehen lassen.“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_loop.hpp"     // 🎯 renderFrame und renderFrame_impl
#include "common.hpp"
#include "settings.hpp"
#include "hud.hpp"
#include "cuda_interop.hpp"

Renderer::Renderer(int width, int height)
    : state(width, height) {}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::initGL() {
#if defined(DEBUG) || defined(_DEBUG)
    if (Settings::debugLogging) std::puts("[DEBUG] initGL aufgerufen");
#endif

    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
#if defined(DEBUG) || defined(_DEBUG)
        std::puts("[ERROR] Fenstererstellung fehlgeschlagen (GLFW)");
#endif
        return;
    }

    if (glewInit() != GLEW_OK) {
#if defined(DEBUG) || defined(_DEBUG)
        std::puts("[ERROR] glewInit() fehlgeschlagen");
#endif
        return;
    }

    RendererWindow::setResizeCallback(state.window, this);
    RendererWindow::setKeyCallback(state.window);

    RendererPipeline::init();

#if defined(DEBUG) || defined(_DEBUG)
    if (Settings::debugLogging) std::puts("[DEBUG] OpenGL-Initialisierung abgeschlossen");
#endif
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::renderFrame(bool autoZoomEnabled) {
    RendererLoop::renderFrame(state, autoZoomEnabled);  // ✅ öffentlich sichtbare Schleife
}

void Renderer::renderFrame_impl(bool autoZoomEnabled) {
    RendererLoop::renderFrame_impl(state, autoZoomEnabled);  // 🔁 interne Schleife bei Bedarf
}

void Renderer::setupBuffers() {
    int totalPixels = state.width * state.height;

    CUDA_CHECK(cudaMalloc(&state.d_iterations, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_entropy, totalPixels * sizeof(float)));

    int tileSize = computeTileSizeFromZoom(state.zoom);
    state.lastTileSize = tileSize;

    int tilesX = state.width / tileSize;
    int tilesY = state.height / tileSize;
    state.h_entropy.resize(tilesX * tilesY);
}

void Renderer::freeDeviceBuffers() {
    if (state.d_iterations) {
        CUDA_CHECK(cudaFree(state.d_iterations));
        state.d_iterations = nullptr;
    }
    if (state.d_entropy) {
        CUDA_CHECK(cudaFree(state.d_entropy));
        state.d_entropy = nullptr;
    }
    state.h_entropy.clear();
}

void Renderer::resize(int newW, int newH) {
    state.width = newW;
    state.height = newH;
    std::printf("[INFO] Resized to %d x %d\n", newW, newH);
}

void Renderer::cleanup() {
    Hud::cleanup();
    RendererPipeline::cleanup();

    // 🔓 CUDA PBO deregistrieren
    CudaInterop::unregisterPBO();

    // 🧹 OpenGL-Ressourcen löschen
    glDeleteBuffers(1, &state.pbo);
    glDeleteTextures(1, &state.tex);

    // 🪟 Fenster schließen
    RendererWindow::destroyWindow(state.window);

    // 🧠 GPU-Speicher freigeben
    freeDeviceBuffers();

    // 🧼 GLFW abschließen
    glfwTerminate();
}

