///// Otter: Progressive-State integriert (d_stateZ/d_stateIt); Only-Grow + Zero-on-Grow.
///  Schneefuchs: Fast-Path prüft jetzt auch Progressive-Buffer; Resize free() ergänzt; ASCII-Logs.
/** Maus: Keine Semantikänderung im Renderpfad – nur Ressourcenverwaltung erweitert; tileSize explizit. */
///// Datei: src/renderer_state.cpp

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"
#include <algorithm>        // std::clamp

// 🦊 Schneefuchs: interne Helfer in anonymer NS, noexcept – klarer Scope.
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
}

RendererState::RendererState(int w, int h)
: width(w), height(h) {
    reset();
}

void RendererState::reset() {
    // Kamera (double)
    zoom   = static_cast<double>(Settings::initialZoom);
    center = make_double2(static_cast<double>(Settings::initialOffsetX),
                          static_cast<double>(Settings::initialOffsetY));
    recomputePixelScale(*this);

    // Iterationen
    baseIterations = Settings::INITIAL_ITERATIONS;
    maxIterations  = Settings::MAX_ITERATIONS_CAP;

    // Timings / Zähler
    fps        = 0.0f;
    deltaTime  = 0.0f;
    frameCount = 0;
    lastTime   = glfwGetTime();

    // Tiles
    lastTileSize = Settings::BASE_TILE_SIZE;

    // Overlays
    heatmapOverlayEnabled       = Settings::heatmapOverlayEnabled;
    warzenschweinOverlayEnabled = Settings::warzenschweinOverlayEnabled;
    warzenschweinText.clear();

    // Host-Analysepuffer
    h_entropy.clear();
    h_contrast.clear();

    // Zoom V3 Silk-Lite
    zoomV3State = {};

    // Progressive: Standard-Schalter aus Settings
    progressiveEnabled = Settings::progressiveDefault;

    // CUDA/Host-Timings: Default (valid=false) + Host-Frame-Nullung
    lastTimings = CudaPhaseTimings{};
    lastTimings.resetHostFrame();
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
    const size_t totalPixels    = size_t(width) * size_t(height);
    int tilesX = 0, tilesY = 0, numTiles = 0;
    computeTiles(width, height, tileSize, tilesX, tilesY, numTiles);

    const size_t it_bytes        = totalPixels * sizeof(int);
    const size_t entropy_bytes   = size_t(numTiles) * sizeof(float);
    const size_t contrast_bytes  = size_t(numTiles) * sizeof(float);
    const size_t progZ_bytes     = totalPixels * sizeof(float2);
    const size_t progIt_bytes    = totalPixels * sizeof(int);

    // 🦦 Otter: Fast-Path – wenn alles schon passt, sofort raus
    const bool sizesOk =
        d_iterations.size() >= it_bytes       &&
        d_entropy.size()    >= entropy_bytes  &&
        d_contrast.size()   >= contrast_bytes &&
        d_stateZ.size()     >= progZ_bytes    &&
        d_stateIt.size()    >= progIt_bytes   &&
        h_entropy.size()    == size_t(numTiles) &&
        h_contrast.size()   == size_t(numTiles) &&
        lastTileSize        == tileSize;

    if (sizesOk) {
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

    // --- iterations buffer (only-grow policy) ---
    {
        const size_t have = d_iterations.size();
        const bool   needGrow = have < it_bytes;
        if (needGrow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_iterations grow: %zu -> %zu", have, it_bytes);
            }
            d_iterations.allocate(it_bytes);
            CUDA_CHECK(cudaMemset(d_iterations.get(), 0, it_bytes));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_iterations ok: have=%zu need=%zu", have, it_bytes);
        }
    }

    // --- entropy buffer (only-grow policy) ---
    {
        const size_t have = d_entropy.size();
        const bool   needGrow = have < entropy_bytes;
        if (needGrow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_entropy grow: %zu -> %zu (tiles %d)", have, entropy_bytes, numTiles);
            }
            d_entropy.allocate(entropy_bytes);
            CUDA_CHECK(cudaMemset(d_entropy.get(), 0, entropy_bytes));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_entropy ok: have=%zu need=%zu (tiles %d)", have, entropy_bytes, numTiles);
        }
    }

    // --- contrast buffer (only-grow policy) ---
    {
        const size_t have = d_contrast.size();
        const bool   needGrow = have < contrast_bytes;
        if (needGrow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_contrast grow: %zu -> %zu (tiles %d)", have, contrast_bytes, numTiles);
            }
            d_contrast.allocate(contrast_bytes);
            CUDA_CHECK(cudaMemset(d_contrast.get(), 0, contrast_bytes));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_contrast ok: have=%zu need=%zu (tiles %d)", have, contrast_bytes, numTiles);
        }
    }

    // --- progressive buffers (only-grow policy) ---
    {
        const size_t haveZ  = d_stateZ.size();
        const size_t haveIt = d_stateIt.size();
        const bool   growZ  = haveZ  < progZ_bytes;
        const bool   growIt = haveIt < progIt_bytes;

        if (growZ) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_stateZ grow: %zu -> %zu", haveZ, progZ_bytes);
            }
            d_stateZ.allocate(progZ_bytes);
            CUDA_CHECK(cudaMemset(d_stateZ.get(), 0, progZ_bytes));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_stateZ ok: have=%zu need=%zu", haveZ, progZ_bytes);
        }

        if (growIt) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_stateIt grow: %zu -> %zu", haveIt, progIt_bytes);
            }
            d_stateIt.allocate(progIt_bytes);
            CUDA_CHECK(cudaMemset(d_stateIt.get(), 0, progIt_bytes));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_stateIt ok: have=%zu need=%zu", haveIt, progIt_bytes);
        }
    }

    if constexpr (Settings::debugLogging) {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaError_t lastErr = cudaGetLastError();
        LUCHS_LOG_HOST("[CHECK] post-alloc sync: err=%d", (int)lastErr);
    }

    // --- host mirror sizes ---
    h_entropy.resize(size_t(numTiles));
    h_contrast.resize(size_t(numTiles));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] buffers ready: it=%p(%zu) entropy=%p(%zu) contrast=%p(%zu) progZ=%p(%zu) progIt=%p(%zu) | %dx%d px, tileSize=%d -> tiles=%d",
                       d_iterations.get(), d_iterations.size(),
                       d_entropy.get(),    d_entropy.size(),
                       d_contrast.get(),   d_contrast.size(),
                       d_stateZ.get(),     d_stateZ.size(),
                       d_stateIt.get(),    d_stateIt.size(),
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

    // Progressive-State freigeben
    d_stateZ.free();
    d_stateIt.free();

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

    // PixelScale hängt von Größe/Zoom ab → neu berechnen
    recomputePixelScale(*this);

    lastTileSize = computeTileSizeFromZoom(static_cast<float>(zoom));
    lastTileSize = std::clamp(lastTileSize, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] resize: zoom=%.5f -> tileSize=%d", zoom, lastTileSize);
    }

    setupCudaBuffers(lastTileSize);

    // Progressive nach Resize: optional eine Ruhe-Frame (sanft) oder State-Zeroing
    progressiveEnabled = Settings::progressiveDefault;
    // (Hard reset ist optional, falls gewünscht:)
    // CUDA_CHECK(cudaMemset(d_stateZ.get(),  0, d_stateZ.size()));
    // CUDA_CHECK(cudaMemset(d_stateIt.get(), 0, d_stateIt.size()));

    // Host-Timings fuer neues Frame auf Null
    lastTimings.resetHostFrame();

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated", width, height);
    }
}
