// Datei: src/renderer_core.cu
// Zeilen: 83
// 🐭 Maus-Kommentar: Entry-Point fürs Rendering. `cleanup()` erfolgt jetzt nur bei erfolgreichem `initGL()` – keine Geisterbefehle mehr bei fehlgeschlagener GL-Init. `resize()` setzt nun korrekt den OpenGL-Viewport. Schneefuchs: „Wer nicht lebt, soll nicht sterben müssen – und wer malt, soll wissen, wo.“

#include "pch.hpp"

#include "renderer_core.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_state.hpp"
#include "renderer_loop.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "hud.hpp"
#include "cuda_interop.hpp"

Renderer::Renderer(int width, int height)
    : state(width, height), glInitialized(false) {}

Renderer::~Renderer() {
    if (glInitialized) {
        cleanup();  // ✅ Nur wenn GL-Kontext erfolgreich initialisiert wurde
    } else if (Settings::debugLogging) {
        std::puts("[DEBUG] cleanup() übersprungen – OpenGL nicht initialisiert");
    }
}

bool Renderer::initGL() {
    if (Settings::debugLogging) std::puts("[DEBUG] initGL aufgerufen");

    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        std::puts("[ERROR] Fenstererstellung fehlgeschlagen (GLFW)");
        return false;
    }

    if (glewInit() != GLEW_OK) {
        std::puts("[ERROR] glewInit() fehlgeschlagen");
        return false;
    }

    // ✅ Callbacks wurden bereits in createWindow(...) gesetzt

    RendererPipeline::init();

    if (Settings::debugLogging) std::puts("[DEBUG] OpenGL-Initialisierung abgeschlossen");
    glInitialized = true;  // 🟢 Flag setzen
    return true;
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::renderFrame(bool autoZoomEnabled) {
    RendererLoop::renderFrame(state, autoZoomEnabled);
}

void Renderer::renderFrame_impl(bool autoZoomEnabled) {
    RendererLoop::renderFrame_impl(state, autoZoomEnabled);
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
    state.resize(newW, newH);

    // 🟢 Viewport korrekt setzen – wichtig nach Fenstergröße-Änderung
    glViewport(0, 0, newW, newH);
}

void Renderer::cleanup() {
    Hud::cleanup();
    RendererPipeline::cleanup();

    CudaInterop::unregisterPBO();

    glDeleteBuffers(1, &state.pbo);
    glDeleteTextures(1, &state.tex);

    RendererWindow::destroyWindow(state.window);

    freeDeviceBuffers();

    glfwTerminate();
}
