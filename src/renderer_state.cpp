///// Otter: Einheitliche, klare Struktur – nur aktive Zustaende; Header schlank, keine PCH; Nacktmull-Pullover.
///// Schneefuchs: Speicher/Buffer exakt definiert; Host-Timings zentral – eine Quelle; /WX-fest; ASCII-only.
///// Maus: Progressive-Defaults aus Settings::progressiveEnabled; Cooldown/State robust.
///// Zaunkönig [ZK]: PBO-Fences sicher aufräumen (resize/Reset), Skip-Flag sauber initialisieren.
///// Datei: src/renderer_state.cpp

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"

#include <algorithm> // clamp, max
#include <cstring>

#include <cuda_runtime_api.h>

// GL forward utils expected in renderer_resources.hpp
// - OpenGLUtils::setGLResourceContext(const char*)
// - OpenGLUtils::createPBO(int w,int h)
// - OpenGLUtils::createTexture(int w,int h)

namespace {
    inline void computeTiles(int width, int height, int tileSize,
                             int& tilesX, int& tilesY, int& numTiles) noexcept {
        tilesX   = (width  + tileSize - 1) / tileSize;
        tilesY   = (height + tileSize - 1) / tileSize;
        numTiles = tilesX * tilesY;
    }

    inline void recomputePixelScale(RendererState& rs) noexcept {
        const double invZoom = (rs.zoom != 0.0) ? (1.0 / rs.zoom) : 1.0;
        const double ar      = (rs.height > 0) ? (double)rs.width / (double)rs.height : 1.0;
        const double sy      = (rs.height > 0) ? (2.0 / (double)rs.height) * invZoom : 2.0 * invZoom;
        rs.pixelScale.y = sy;
        rs.pixelScale.x = sy * ar;
    }

    inline void clearPboFences(RendererState& rs) noexcept {
        OpenGLUtils::setGLResourceContext("pbo-fence-clear");
        for (auto& f : rs.pboFence) {
            if (f) { glDeleteSync(f); f = 0; }
        }
    }
}

// =============================== Streams / Events =============================

void RendererState::createCudaStreamsIfNeeded() {
    CUDA_CHECK(cudaSetDevice(0));
    if (!renderStream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&renderStream, cudaStreamNonBlocking));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[STREAM] renderStream created %p (non-blocking)", (void*)renderStream);
        }
    }
    if (!copyStream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[STREAM] copyStream created %p (non-blocking)", (void*)copyStream);
        }
    }
}

void RendererState::destroyCudaStreamsIfAny() noexcept {
    if (renderStream) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[STREAM] renderStream destroy %p", (void*)renderStream);
        }
        cudaStreamDestroy(renderStream);
        renderStream = nullptr;
    }
    if (copyStream) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[STREAM] copyStream destroy %p", (void*)copyStream);
        }
        cudaStreamDestroy(copyStream);
        copyStream = nullptr;
    }
}

void RendererState::createCudaEventsIfNeeded() {
    if (!evEcDone) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evEcDone, cudaEventDisableTiming));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EVENT] evEcDone created %p (disableTiming)", (void*)evEcDone);
        }
    }
    if (!evCopyDone) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evCopyDone, cudaEventDisableTiming));
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EVENT] evCopyDone created %p (disableTiming)", (void*)evCopyDone);
        }
    }
}

void RendererState::destroyCudaEventsIfAny() noexcept {
    if (evEcDone) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EVENT] evEcDone destroy %p", (void*)evEcDone);
        }
        cudaEventDestroy(evEcDone);
        evEcDone = nullptr;
    }
    if (evCopyDone) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[EVENT] evCopyDone destroy %p", (void*)evCopyDone);
        }
        cudaEventDestroy(evCopyDone);
        evCopyDone = nullptr;
    }
}

// =============================== Host Pinning =================================

void RendererState::ensureHostPinnedForAnalysis() {
    // Register host vectors as pinned (idempotent). Must be sized already.
    if (!h_entropy.empty() && !h_entropyPinned) {
        CUDA_CHECK(cudaHostRegister(h_entropy.data(), h_entropy.size() * sizeof(float), cudaHostRegisterDefault));
        h_entropyPinned = true;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] h_entropy pinned ptr=%p bytes=%zu",
                           (void*)h_entropy.data(), h_entropy.size() * sizeof(float));
        }
    }
    if (!h_contrast.empty() && !h_contrastPinned) {
        CUDA_CHECK(cudaHostRegister(h_contrast.data(), h_contrast.size() * sizeof(float), cudaHostRegisterDefault));
        h_contrastPinned = true;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] h_contrast pinned ptr=%p bytes=%zu",
                           (void*)h_contrast.data(), h_contrast.size() * sizeof(float));
        }
    }
}

void RendererState::unpinHostAnalysisIfAny() noexcept {
    if (h_entropyPinned && !h_entropy.empty()) {
        cudaHostUnregister(h_entropy.data());
        h_entropyPinned = false;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] h_entropy unpinned ptr=%p", (void*)h_entropy.data());
        }
    }
    if (h_contrastPinned && !h_contrast.empty()) {
        cudaHostUnregister(h_contrast.data());
        h_contrastPinned = false;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIN] h_contrast unpinned ptr=%p", (void*)h_contrast.data());
        }
    }
}

// ================================== Ctor/Dtor =================================

RendererState::RendererState(int w, int h)
: width(w), height(h) {
    createCudaStreamsIfNeeded();
    createCudaEventsIfNeeded();
    reset();
}

RendererState::~RendererState() {
    clearPboFences(*this);
    unpinHostAnalysisIfAny();
    destroyCudaEventsIfAny();
    destroyCudaStreamsIfAny();
}

// =================================== Reset ===================================

void RendererState::reset() {
    zoom   = static_cast<double>(Settings::initialZoom);
    center = make_double2(static_cast<double>(Settings::initialOffsetX),
                          static_cast<double>(Settings::initialOffsetY));
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

    // Host mirrors (unpinned first to avoid stale registrations after resize)
    unpinHostAnalysisIfAny();
    h_entropy.clear();
    h_contrast.clear();

    // Zoom V3 state clean
    zoomV3State = {};

    // Progressive defaults from Settings
    progressiveEnabled         = Settings::progressiveEnabled;
    progressiveCooldownFrames  = 0;

    // Zaunkönig: fences & upload flag
    skipUploadThisFrame = false;
    clearPboFences(*this);

    lastTimings = CudaPhaseTimings{};
    lastTimings.resetHostFrame();

    // Ensure infra is present
    createCudaStreamsIfNeeded();
    createCudaEventsIfNeeded();
}

// ================================ CUDA Buffers ================================

void RendererState::setupCudaBuffers(int tileSize) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] setupCudaBuffers: invalid size %dx%d", width, height);
        }
        return;
    }
    tileSize = std::clamp(tileSize, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);

    const size_t totalPixels    = size_t(width) * size_t(height);
    int tilesX = 0, tilesY = 0, numTiles = 0;
    computeTiles(width, height, tileSize, tilesX, tilesY, numTiles);

    const size_t it_bytes       = totalPixels * sizeof(uint16_t);
    const size_t entropy_bytes  = size_t(numTiles) * sizeof(float);
    const size_t contrast_bytes = size_t(numTiles) * sizeof(float);

    const bool   wantProg       = Settings::progressiveEnabled;
    const size_t z_bytes        = totalPixels * sizeof(float2);
    const size_t it2_bytes      = totalPixels * sizeof(uint16_t);

    const bool sizesOk =
        d_iterations.size() >= it_bytes      &&
        d_entropy.size()    >= entropy_bytes &&
        d_contrast.size()   >= contrast_bytes &&
        (!wantProg || (d_stateZ.size() >= z_bytes && d_stateIt.size() >= it2_bytes)) &&
        h_entropy.size()    == size_t(numTiles) &&
        h_contrast.size()   == size_t(numTiles) &&
        lastTileSize        == tileSize;

    if (sizesOk) {
        // Ensure pinning if vectors were reallocated elsewhere
        ensureHostPinnedForAnalysis();
        return;
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] setupCudaBuffers: w=%d h=%d zoom=%.5f tileSize=%d tiles=%d (%d x %d) pixels=%zu",
                       width, height, zoom, tileSize, numTiles, tilesX, tilesY, totalPixels);
    }

    CUDA_CHECK(cudaSetDevice(0));
    if constexpr (Settings::debugLogging) {
        CudaInterop::logCudaDeviceContext("setupCudaBuffers");
    }

    // Device buffers
    {
        // d_iterations
        const size_t have = d_iterations.size();
        const bool   grow = have < it_bytes;
        if (grow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_iterations grow: %zu -> %zu", have, it_bytes);
            }
            d_iterations.allocate(it_bytes);
        }
        // d_entropy
        const size_t haveE = d_entropy.size();
        const bool   growE = haveE < entropy_bytes;
        if (growE) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_entropy grow: %zu -> %zu (tiles %d)", haveE, entropy_bytes, numTiles);
            }
            d_entropy.allocate(entropy_bytes);
        }
        // d_contrast
        const size_t haveC = d_contrast.size();
        const bool   growC = haveC < contrast_bytes;
        if (growC) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_contrast grow: %zu -> %zu (tiles %d)", haveC, contrast_bytes, numTiles);
            }
            d_contrast.allocate(contrast_bytes);
        }

        // progressive
        if (wantProg) {
            const size_t haveZ  = d_stateZ.size();
            const bool   growZ  = haveZ < z_bytes;
            if (growZ) {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ALLOC] d_stateZ grow: %zu -> %zu (px %zu)", haveZ, z_bytes, totalPixels);
                }
                d_stateZ.allocate(z_bytes);
            }
            const size_t haveIt = d_stateIt.size();
            const bool   growIt = haveIt < it2_bytes;
            if (growIt) {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ALLOC] d_stateIt grow: %zu -> %zu (px %zu)", haveIt, it2_bytes, totalPixels);
                }
                d_stateIt.allocate(it2_bytes);
            }
        }

        if constexpr (Settings::debugLogging) {
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaError_t lastErr = cudaGetLastError();
            LUCHS_LOG_HOST("[CHECK] post-alloc sync: err=%d", (int)lastErr);
        }
    }

    // Host mirrors (reserve for MIN_TILE_SIZE worst case, then size to current)
    {
        const int tilesXmax = (width  + Settings::MIN_TILE_SIZE - 1) / Settings::MIN_TILE_SIZE;
        const int tilesYmax = (height + Settings::MIN_TILE_SIZE - 1) / Settings::MIN_TILE_SIZE;
        const size_t numTilesMax = size_t(tilesXmax) * size_t(tilesYmax);

        // Unregister before vector might reallocate
        unpinHostAnalysisIfAny();

        if (h_entropy.capacity()  < numTilesMax)  h_entropy.reserve(numTilesMax);
        if (h_contrast.capacity() < numTilesMax)  h_contrast.reserve(numTilesMax);
        h_entropy.resize(size_t(numTiles));
        h_contrast.resize(size_t(numTiles));

        // Re-register pinned after (re)allocation
        ensureHostPinnedForAnalysis();
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] buffers ready: it=%p(%zu) entropy=%p(%zu) contrast=%p(%zu) z=%p(%zu) it2=%p(%zu) | %dx%d px, tileSize=%d -> tiles=%d",
                       d_iterations.get(), d_iterations.size(),
                       d_entropy.get(),    d_entropy.size(),
                       d_contrast.get(),   d_contrast.size(),
                       Settings::progressiveEnabled ? d_stateZ.get()  : nullptr, Settings::progressiveEnabled ? d_stateZ.size()  : 0,
                       Settings::progressiveEnabled ? d_stateIt.get() : nullptr, Settings::progressiveEnabled ? d_stateIt.size() : 0,
                       width, height, tileSize, numTiles);
    }

    lastTileSize = tileSize;
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
    d_entropy.free();
    d_contrast.free();
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
    for (auto& b : pboRing) { b = Hermelin::GLBuffer(OpenGLUtils::createPBO(width, height)); }
    pboIndex = 0;
    std::fill(pboFence.begin(), pboFence.end(), (GLsync)0);
    skipUploadThisFrame = false;
    tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(width, height));

    { GLuint ids[kPboRingSize] = { pboRing[0].id(), pboRing[1].id(), pboRing[2].id() };
      CudaInterop::registerAllPBOs(ids, kPboRingSize); }

    recomputePixelScale(*this);

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));
    lastTileSize = std::clamp(lastTileSize, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] resize: zoom=%.5f -> tileSize=%d", zoom, lastTileSize);
    }

    setupCudaBuffers(lastTileSize);
    lastTimings.resetHostFrame();

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated", width, height);
    }
}

// ========================= Progressive State Control ==========================

void RendererState::invalidateProgressiveState(bool hardReset) noexcept {
    progressiveEnabled        = false;
    progressiveCooldownFrames = 2;

    if (hardReset) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PROG] hardReset requested (state will be cleared on next allocation)");
        }
    } else {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PROG] soft invalidate: cooldown=%d", progressiveCooldownFrames);
        }
    }
}
