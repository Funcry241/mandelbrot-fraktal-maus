// MAUS:
// Datei: src/renderer_state.cpp
// 🐭 Maus: Sichtbare Tile-/Buffer-Sanity, kein „malloc into the void“, deterministische Logs.
// 🦦 Otter: Explizite Realloc-Policy (only-grow) und klare Lebenszyklen von CUDA/GL-Ressourcen. (Bezug zu Otter)
// 🦊 Schneefuchs: Keine impliziten Seiteneffekte, Header/Source konsistent, ASCII-only Logs. (Bezug zu Schneefuchs)

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"
#include "opengl_utils.hpp" // Schneefuchs: explizit, da setGLResourceContext / PBO/Texture genutzt
#include <algorithm>        // std::clamp

// Helper: compute tile layout for given tileSize
static inline void computeTiles(int width, int height, int tileSize,
                                int& tilesX, int& tilesY, int& numTiles) {
    tilesX   = (width  + tileSize - 1) / tileSize;
    tilesY   = (height + tileSize - 1) / tileSize;
    numTiles = tilesX * tilesY;
}

RendererState::RendererState(int w, int h)
: width(w), height(h) {
    reset();
}

void RendererState::reset() {
    // Camera
    zoom   = static_cast<double>(Settings::initialZoom);
    offset = { static_cast<float>(Settings::initialOffsetX), static_cast<float>(Settings::initialOffsetY) };

    // Iterations
    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations  = Settings::MAX_ITERATIONS_CAP;

    // Timers / Counters
    fps        = 0.0f;
    deltaTime  = 0.0f;
    frameCount = 0;
    lastTime   = glfwGetTime();

    // Tiles
    lastTileSize = Settings::BASE_TILE_SIZE;

    // Overlays
    heatmapOverlayEnabled       = Settings::heatmapOverlayEnabled;
    warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;

    // Host analysis buffers
    h_entropy.clear();
    h_contrast.clear();

    // Zoom V2 state (explicit reset – no globals)
    zoomV2State = ZoomLogic::ZoomState{};

    // CUDA timings reset
    lastTimings = CudaPhaseTimings{}; // valid=false by default
}

void RendererState::setupCudaBuffers(int tileSize) {
    // --- sanitize input ---
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] setupCudaBuffers: invalid size %dx%d", width, height);
        }
        return;
    }
    // Clamp tile size to configured bounds
    tileSize = std::clamp(tileSize, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);

    // --- derive sizes ---
    const int totalPixels = width * height;
    int tilesX = 0, tilesY = 0, numTiles = 0;
    computeTiles(width, height, tileSize, tilesX, tilesY, numTiles);

    const size_t it_bytes       = static_cast<size_t>(totalPixels) * sizeof(int);
    const size_t entropy_bytes  = static_cast<size_t>(numTiles)    * sizeof(float);
    const size_t contrast_bytes = static_cast<size_t>(numTiles)    * sizeof(float);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] setupCudaBuffers: w=%d h=%d zoom=%.5f tileSize=%d tiles=%d (%d x %d) pixels=%d",
                       width, height, zoom, tileSize, numTiles, tilesX, tilesY, totalPixels);
    }

    CUDA_CHECK(cudaSetDevice(0));
    if constexpr (Settings::debugLogging) {
        CudaInterop::logCudaDeviceContext("setupCudaBuffers");
    }

    // --- iterations buffer (only-grow policy) ---
    const size_t have_it = d_iterations.size();
    if (have_it < it_bytes) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_iterations grow: have=%zu -> need=%zu", have_it, it_bytes);
        }
        d_iterations.allocate(it_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_iterations ok: have=%zu need=%zu (no realloc)", have_it, it_bytes);
    }
    CUDA_CHECK(cudaMemset(d_iterations.get(), 0, it_bytes));
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CHECK] cudaMemset d_iterations ok (bytes=%zu)", it_bytes);
    }

    // --- entropy buffer (only-grow policy) ---
    const size_t have_entropy = d_entropy.size();
    if (have_entropy < entropy_bytes) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_entropy grow: have=%zu -> need=%zu (tiles %d)", have_entropy, entropy_bytes, numTiles);
        }
        d_entropy.allocate(entropy_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_entropy ok: have=%zu need=%zu (tiles %d, no realloc)", have_entropy, entropy_bytes, numTiles);
    }

    // Diagnostics: pointer attributes (ASCII only)
    if constexpr (Settings::debugLogging) {
        cudaPointerAttributes attr = {};
        cudaError_t attrErr = cudaPointerGetAttributes(&attr, d_entropy.get());
        LUCHS_LOG_HOST("[CHECK] d_entropy attr: err=%d type=%d device=%d hostPtr=%p devicePtr=%p",
                       (int)attrErr, (int)attr.type, (int)attr.device,
                       (void*)attr.hostPointer, (void*)attr.devicePointer);
    }

    CUDA_CHECK(cudaMemset(d_entropy.get(), 0, entropy_bytes));
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CHECK] cudaMemset d_entropy ok (bytes=%zu)", entropy_bytes);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    {
        cudaError_t syncErr = cudaGetLastError();
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CHECK] post-entropy sync: err=%d", (int)syncErr);
        }
        if (syncErr != cudaSuccess)
            throw std::runtime_error("cudaMemset d_entropy failed (post-sync)");
    }

    // --- contrast buffer (only-grow policy) ---
    const size_t have_contrast = d_contrast.size();
    if (have_contrast < contrast_bytes) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_contrast grow: have=%zu -> need=%zu (tiles %d)", have_contrast, contrast_bytes, numTiles);
        }
        d_contrast.allocate(contrast_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_contrast ok: have=%zu need=%zu (tiles %d, no realloc)", have_contrast, contrast_bytes, numTiles);
    }
    CUDA_CHECK(cudaMemset(d_contrast.get(), 0, contrast_bytes));
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CHECK] cudaMemset d_contrast ok (bytes=%zu)", contrast_bytes);
    }

    // --- host mirror sizes ---
    h_entropy.resize(numTiles);
    h_contrast.resize(numTiles);

    // --- summary ---
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] buffers ready: it=%p(%zu) entropy=%p(%zu) contrast=%p(%zu) | %dx%d px, tileSize=%d -> tiles=%d",
                       d_iterations.get(), d_iterations.size(),
                       d_entropy.get(),    d_entropy.size(),
                       d_contrast.get(),   d_contrast.size(),
                       width, height, tileSize, numTiles);
    }

    lastTileSize = tileSize;
}

void RendererState::resize(int newWidth, int newHeight) {
    if (newWidth <= 0 || newHeight <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] resize: invalid target size %d x %d", newWidth, newHeight);
        }
        return;
    }

    // Free old CUDA device buffers
    d_iterations.free();
    d_entropy.free();
    d_contrast.free();

    // Unregister CUDA-GL PBO
    CudaInterop::unregisterPBO();

    // Free GL buffers via RAII
    pbo.free();
    tex.free();

    // Apply new size
    width  = newWidth;
    height = newHeight;

    OpenGLUtils::setGLResourceContext("resize");

    // Create fresh GL buffers
    pbo = Hermelin::GLBuffer(OpenGLUtils::createPBO(width, height));
    tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(width, height));

    CudaInterop::registerPBO(pbo);

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));
    lastTileSize = std::clamp(lastTileSize, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] resize: zoom=%.5f -> tileSize=%d", zoom, lastTileSize);
    }

    setupCudaBuffers(lastTileSize);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[Resize] %d x %d buffers reallocated", width, height);
    }
}
