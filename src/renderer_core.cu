// Datei: src/renderer_core.cu
// Zeilen: 78
// 🐭 Maus-Kommentar: Entry-Point fürs Rendering. Entfernt: ungenutztes `setupBuffers()`. Cleanup durch Destruktor bleibt. Schneefuchs: „Weniger ist mehr – wenn der Code schweigt, wird der Otter klug.“

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
    if (Settings::debugLogging) std::puts("[DEBUG] initGL aufgerufen");

    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        std::puts("[ERROR] Fenstererstellung fehlgeschlagen (GLFW)");
        return;
    }

    if (glewInit() != GLEW_OK) {
        std::puts("[ERROR] glewInit() fehlgeschlagen");
        return;
    }

    RendererWindow::setResizeCallback(state.window, this);
    RendererWindow::setKeyCallback(state.window);

    RendererPipeline::init();

    if (Settings::debugLogging) std::puts("[DEBUG] OpenGL-Initialisierung abgeschlossen");
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
    std::printf("[INFO] Resized to %d x %d\n", newW, newH);
    state.resize(newW, newH);  // ✅ führt vollständiges Reset samt PBO/CUDA durch
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
