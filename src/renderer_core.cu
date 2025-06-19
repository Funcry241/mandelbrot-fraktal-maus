// Datei: src/renderer_core.cu
// Zeilen: 67
// 🐭 Maus-Kommentar: Entry-Point für das Rendering-Modul. `RendererWindow` managt Fenster, `RendererPipeline` initialisiert. Neu: `renderFrame()` ruft korrekt `RendererLoop::renderFrame(...)` auf (nicht impl direkt!). Schneefuchs: „Kein Shortcut durch den Sumpf!“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_loop.hpp"     // 🎯 renderFrame und renderFrame_impl
#include "common.hpp"

Renderer::Renderer(int width, int height)
    : state(width, height) {}

Renderer::~Renderer() {}

void Renderer::initGL() {
    state.window = RendererWindow::createWindow(state.width, state.height, this);
    RendererWindow::setResizeCallback(state.window, this);
    RendererWindow::setKeyCallback(state.window);
    RendererPipeline::init();  // ✅ keine Parameter mehr
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

    int tilesX = state.width / state.lastTileSize;
    int tilesY = state.height / state.lastTileSize;
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
