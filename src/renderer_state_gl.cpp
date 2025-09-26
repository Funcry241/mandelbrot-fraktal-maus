///// Otter: Split – GL-Fences, PBO-Ring, Resize, Reset & Dtor; saubere Ring-Disziplin.
///// Schneefuchs: EC-Pfade entfernt (keine Host-Mirror/Pinning mehr); GLsync-Abräumung zentral; /WX-fest.
///// Maus: PixelScale-Recompute lokal; Events/Streams via CUDA-TU; Logs ASCII-only; unter 300 Zeilen.
///// Datei: src/renderer_state_gl.cpp

#include "pch.hpp"
#include "luchs_log_host.hpp"
#include <GL/glew.h>

#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"
#include "zoom_logic.hpp"   // <- für computeTileSizeFromZoom

#include <algorithm>

namespace {
// ----- PixelScale (GL-Seite nutzt Reset/Resize) --------------------------------
// Korrektur: PixelScale ist **zoomfrei** und **isotrop** (x==y). Das Seitenverhältnis
// entsteht automatisch über width/height in den Pixel-Offsets. Kein ar-Scaling hier.
inline void recomputePixelScale(RendererState& rs) noexcept {
    const double sy = (rs.height > 0) ? (2.0 / static_cast<double>(rs.height)) : 2.0;
    rs.pixelScale.y = sy;
    rs.pixelScale.x = sy; // isotrop; kein ar, kein 1/zoom
}

inline void clearPboFences(RendererState& rs) noexcept {
    OpenGLUtils::setGLResourceContext("pbo-fence-clear");
    for (auto& f : rs.pboFence) {
        if (f) { glDeleteSync(f); f = 0; }
    }
}
} // namespace

// ================================== Ctor/Dtor =================================

RendererState::~RendererState() {
    clearPboFences(*this);
    // GL: Ring + Texture freigeben (falls noch vorhanden)
    CudaInterop::unregisterAllPBOs();
    for (auto& b : pboRing) { b.free(); }
    tex.free();

    // CUDA: Streams/Events
    destroyCudaEventsIfAny();
    destroyCudaStreamsIfAny();
}

// =================================== Reset ===================================

void RendererState::reset() {
    zoom   = static_cast<double>(Settings::initialZoom);
    center = double2{ static_cast<double>(Settings::initialOffsetX),
                      static_cast<double>(Settings::initialOffsetY) };
    recomputePixelScale(*this);

    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations  = Settings::MAX_ITERATIONS_CAP;

    fps        = 0.0f;
    deltaTime  = 0.0f;
    frameCount = 0;
    lastTime   = glfwGetTime();

    lastTileSize = Settings::BASE_TILE_SIZE;

    heatmapOverlayEnabled       = Settings::heatmapOverlayEnabled;
    warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;
    warzenschweinText.clear();

    // Zoom V3 state clean
    zoomV3State = {};

    // Progressive defaults
    progressiveEnabled         = Settings::progressiveEnabled;
    progressiveCooldownFrames  = 0;

    // Zaunkönig: fences & upload flag & ring stats
    skipUploadThisFrame = false;
    clearPboFences(*this);
    pboIndex = 0;
    std::fill(ringUse.begin(), ringUse.end(), 0u);
    ringSkip = 0;

    lastTimings = CudaPhaseTimings{};
    lastTimings.resetHostFrame();

    // Ensure CUDA infra is present
    createCudaStreamsIfNeeded();
    createCudaEventsIfNeeded();
}

// ================================== Resize ===================================

void RendererState::resize(int newWidth, int newHeight) {
    if (newWidth <= 0 || newHeight <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] resize: invalid target size %d x %d", newWidth, newHeight);
        }
        return;
    }

    // GL / CUDA teardown for old size
    clearPboFences(*this);

    d_iterations.free();
    d_stateZ.free();
    d_stateIt.free();

    CudaInterop::unregisterAllPBOs();

    for (auto& b : pboRing) { b.free(); }
    tex.free();

    // Apply new size
    width  = newWidth;
    height = newHeight;

    // Recreate GL side
    OpenGLUtils::setGLResourceContext("resize");
    for (auto& b : pboRing) {
        b = Hermelin::GLBuffer(OpenGLUtils::createPBO(width, height));
    }

    pboIndex = 0;
    std::fill(pboFence.begin(), pboFence.end(), (GLsync)0);
    skipUploadThisFrame = false;

    tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(width, height));

    // Dynamisch alle PBO-IDs sammeln und registrieren (Ringgröße = kPboRingSize)
    {
        unsigned int ids[RendererState::kPboRingSize];
        for (int i = 0; i < RendererState::kPboRingSize; ++i) {
            ids[i] = pboRing[i].id();
        }
        CudaInterop::registerAllPBOs(ids, RendererState::kPboRingSize);
    }

    recomputePixelScale(*this);

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));
    lastTileSize = std::clamp(lastTileSize, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] resize: zoom=%.5f -> tileSize=%d", zoom, lastTileSize);
    }

    setupCudaBuffers(lastTileSize);
    lastTimings.resetHostFrame();

    // Ring-Statistik zum neuen Start nullen (LOG-6)
    std::fill(ringUse.begin(), ringUse.end(), 0u);
    ringSkip = 0;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated", width, height);
    }
}
