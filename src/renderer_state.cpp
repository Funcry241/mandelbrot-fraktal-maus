// 🐭 Maus: Sichtbare Tile-/Buffer-Sanity, kein "malloc into the void", deterministische Logs.
// 🦦 Otter: Only-Grow + Fast-Path ohne Memsets/Sync bei unveränderten Größen. (Bezug zu Otter)
// 🦊 Schneefuchs: Host-Timings zentral (resetHostFrame), ASCII-only Logs, /WX-fest. (Bezug zu Schneefuchs)
// 🐑 Schneefuchs: Teures vermeiden #2 – Host-Vectoren wachsen in Potenzen von 2 (weniger Realloc/Copy).

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"
#include "opengl_utils.hpp" // setGLResourceContext / PBO/Texture
#include <algorithm>        // std::clamp
#include <cstdint>

// ---- Host-Timings: Definition der Methodik aus dem Header --------------------
void RendererState::CudaPhaseTimings::resetHostFrame() noexcept {
    uploadMs     = 0.0;
    overlaysMs   = 0.0;
    frameTotalMs = 0.0;
}

// Helper: compute tile layout for given tileSize (header-local, ODR-safe)
static inline void computeTiles(int width, int height, int tileSize,
                                int& tilesX, int& tilesY, int& numTiles) {
    tilesX   = (width  + tileSize - 1) / tileSize;
    tilesY   = (height + tileSize - 1) / tileSize;
    numTiles = tilesX * tilesY;
}

// 🐑 Schneefuchs: next power-of-two (32-bit) – reduziert Reallocs beim Zoom
static inline size_t nextPow2_u32(size_t v) {
    if (v <= 1) return 1;
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    v |= v >> 32; // 64-bit safety
#endif
    return v + 1;
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
    h_entropy.shrink_to_fit();
    h_contrast.shrink_to_fit();

    // Zoom V2 state (explicit reset – no globals)
    zoomV2State = ZoomLogic::ZoomState{};

    // Timings reset (CUDA + HOST auf Default)
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

    // 🦦 Otter: Fast-Path – wenn alles schon passt, sofort raus (keine Memsets/Sync)
    const bool sizesOk =
        d_iterations.size() >= it_bytes   &&
        d_entropy.size()    >= entropy_bytes &&
        d_contrast.size()   >= contrast_bytes &&
        h_entropy.size()    == static_cast<size_t>(numTiles) &&
        h_contrast.size()   == static_cast<size_t>(numTiles) &&
        lastTileSize        == tileSize;

    if (sizesOk) {
        return;
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] setupCudaBuffers: w=%d h=%d zoom=%.5f tileSize=%d tiles=%d (%d x %d) pixels=%d",
                       width, height, zoom, tileSize, numTiles, tilesX, tilesY, totalPixels);
    }

    // 🐑 Schneefuchs: Device nur einmal setzen (teuren Call nicht jede Frame)
    static bool s_deviceBound = false;
    if (!s_deviceBound) {
        CUDA_CHECK(cudaSetDevice(0));
        s_deviceBound = true;
        if constexpr (Settings::debugLogging) {
            CudaInterop::logCudaDeviceContext("setupCudaBuffers");
        }
    }

    // --- iterations buffer (only-grow policy) ---
    const size_t have_it = d_iterations.size();
    const bool   grow_it = have_it < it_bytes;
    if (grow_it) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_iterations grow: have=%zu -> need=%zu", have_it, it_bytes);
        }
        d_iterations.allocate(it_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_iterations ok: have=%zu need=%zu (no realloc)", have_it, it_bytes);
    }

    // --- entropy buffer (only-grow policy) ---
    const size_t have_entropy = d_entropy.size();
    const bool   grow_entropy = have_entropy < entropy_bytes;
    if (grow_entropy) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_entropy grow: have=%zu -> need=%zu (tiles %d)", have_entropy, entropy_bytes, numTiles);
        }
        d_entropy.allocate(entropy_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_entropy ok: have=%zu need=%zu (tiles %d, no realloc)", have_entropy, entropy_bytes, numTiles);
    }

    // --- contrast buffer (only-grow policy) ---
    const size_t have_contrast = d_contrast.size();
    const bool   grow_contrast = have_contrast < contrast_bytes;
    if (grow_contrast) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_contrast grow: have=%zu -> need=%zu (tiles %d)", have_contrast, contrast_bytes, numTiles);
        }
        d_contrast.allocate(contrast_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_contrast ok: have=%zu need=%zu (tiles %d, no realloc)", have_contrast, contrast_bytes, numTiles);
    }

    // 🐭 Maus: Zeroing nur bei NEU-Allocation (oder in Debug immer) → spart Bandbreite & Stalls
    const bool clearIterations = grow_it       || Settings::debugLogging;
    const bool clearEntropy    = grow_entropy  || Settings::debugLogging;
    const bool clearContrast   = grow_contrast || Settings::debugLogging;

    if (clearIterations) CUDA_CHECK(cudaMemset(d_iterations.get(), 0, it_bytes));
    if (clearEntropy)    CUDA_CHECK(cudaMemset(d_entropy.get(),    0, entropy_bytes));
    if (clearContrast)   CUDA_CHECK(cudaMemset(d_contrast.get(),   0, contrast_bytes));

    // 🐑 Schneefuchs: optionaler Debug-Sync zur klaren Fehlerlokalisierung
    if constexpr (Settings::debugLogging) {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaError_t syncErr = cudaGetLastError();
        LUCHS_LOG_HOST("[CHECK] post-alloc sync: err=%d", (int)syncErr);
    }

    // --- host mirror sizes (POT-capacity -> weniger teure Realloc/Copy bei Zoom-Drift) ---
    auto ensureHostVec = [&](auto& vec) {
        const size_t need = static_cast<size_t>(numTiles);
        if (vec.capacity() < need) {
            const size_t newCap = nextPow2_u32(need);
            vec.reserve(newCap);
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] host reserve: cap=%zu -> %zu (need=%zu)", vec.capacity(), newCap, need);
            }
        }
        vec.resize(need);
    };
    ensureHostVec(h_entropy);
    ensureHostVec(h_contrast);

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

    // Host-Timings fuer neues Frame auf Null
    lastTimings.resetHostFrame();

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated", width, height);
    }
}
