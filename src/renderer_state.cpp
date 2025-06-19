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

    smoothedZoom = zoom;
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

void RendererState::applyLerpStep() {
    smoothedZoom += (targetZoom - smoothedZoom) * Settings::LERP_FACTOR;
    smoothedOffset.x += (targetOffset.x - smoothedOffset.x) * Settings::LERP_FACTOR;
    smoothedOffset.y += (targetOffset.y - smoothedOffset.y) * Settings::LERP_FACTOR;

    zoom = smoothedZoom;
    offset = smoothedOffset;
}

void RendererState::adaptIterationCount() {
    float logZoom = std::log10(zoom);
    maxIterations = static_cast<int>(baseIterations + logZoom * 200.0f);
    maxIterations = std::min(maxIterations, Settings::MAX_ITERATIONS_CAP);
}

// 🧰 Initialisiert OpenGL-Textur, PBO, registriert CUDA-Zugriff und allokiert Device-Puffer
void initResources(RendererState& state) {
    // 🔧 OpenGL-Textur anlegen
    glGenTextures(1, &state.tex);
    glBindTexture(GL_TEXTURE_2D, state.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, state.width, state.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // 🔧 PBO erzeugen
    glGenBuffers(1, &state.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, state.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, state.width * state.height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 🔗 CUDA/OpenGL Interop
    CudaInterop::registerPBO(state.pbo);

    // ⚡ CUDA Device-Memory allokieren
    size_t numPixels = static_cast<size_t>(state.width) * static_cast<size_t>(state.height);
    CUDA_CHECK(cudaMalloc(&state.d_iterations, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_entropy, numPixels * sizeof(float)));

    // 🧠 Host-Speicher vorbereiten
    state.h_entropy.resize(numPixels);
}
