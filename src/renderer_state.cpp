// Datei: src/renderer_state.cpp
// Zeilen: 64
// 🐭 Maus-Kommentar: Kapselt Zustand & Steuerung des Fraktal-Renderers. Zwei reset()-Methoden waren definiert – jetzt korrekt konsolidiert. `lastTime` und `lastFrameTime` setzen die Framerate-Messung zurück. Schneefuchs: „Doppelte Definition ist wie doppelt gezoomt: verwirrend und unnötig.“

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"

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
