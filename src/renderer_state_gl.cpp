///// Otter: Split - GL-Fences, PBO- & TEX-Ring; Resize/Reset & Dtor; draw-lag-1 vorbereitet
///// Schneefuchs: EC-Pfade entfernt; GLsync-Abräumung zentral; DSA-freundliche PixelStore-Policy
///// Maus: PixelScale zoomfrei/isotrop; ASCII-Logs; kompakt
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
// Zoomfrei & isotrop (x==y). Seitenverhältnis ergibt sich aus width/height bei Pixel-Offsets.
inline void recomputePixelScale(RendererState& rs) noexcept {
    const double sy = (rs.height > 0) ? (2.0 / static_cast<double>(rs.height)) : 2.0;
    rs.pixelScale.y = sy;
    rs.pixelScale.x = sy;
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
    // GL: PBO-Ring + Textur-Ring freigeben
    CudaInterop::unregisterAllPBOs();
    for (auto& b : pboRing) { b.free(); }
    for (auto& t : texRing) { t.free(); }
    tex.free(); // legacy (id==0, falls nie verwendet)

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

    // Texture-Ring-Indizes
    texUploadIndex = 0;
    texDrawIndex   = (kTexRingSize + texUploadIndex - 1) % kTexRingSize; // initial draw-lag-1

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
    for (auto& t : texRing) { t.free(); }
    tex.free(); // legacy

    // Apply new size
    width  = newWidth;
    height = newHeight;

    // DSA-freundliche, deterministische PixelStore-Policy einmal setzen
    glPixelStorei(GL_UNPACK_ALIGNMENT,   1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH,  0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS,   0);

    // Recreate GL side
    OpenGLUtils::setGLResourceContext("resize");
    for (auto& b : pboRing) {
        b = Hermelin::GLBuffer(OpenGLUtils::createPBO(width, height));
    }

    pboIndex = 0;
    std::fill(pboFence.begin(), pboFence.end(), (GLsync)0);
    skipUploadThisFrame = false;

    for (auto& t : texRing) {
        t = Hermelin::GLBuffer(OpenGLUtils::createTexture(width, height));
    }
    // Legacy-Einzeltextur bleibt 0 (ungültig), um Doppel-Frees zu vermeiden.

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

    // Texture-Ring-Indizes resetten (draw-lag-1)
    texUploadIndex = 0;
    texDrawIndex   = (kTexRingSize + texUploadIndex - 1) % kTexRingSize;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated (PBO=%d TEX=%d)",
                       width, height, (int)kPboRingSize, (int)kTexRingSize);
    }
}
