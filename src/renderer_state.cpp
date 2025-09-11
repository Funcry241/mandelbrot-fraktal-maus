///// Otter: Einheitliche, klare Struktur – nur aktive Zustaende; Header schlank, keine PCH; Nacktmull-Pullover.
///// Schneefuchs: Speicher/Buffer exakt definiert; Host-Timings zentral – eine Quelle; /WX-fest; ASCII-only.
///// Maus: tileSize bleibt in Pipelines explizit; hier nur Zustand & Ressourcen; keine versteckten Semantikwechsel.
///// Datei: src/renderer_state.cpp

#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"
#include <algorithm>        // std::clamp
#include <GLFW/glfw3.h>     // glfwGetTime

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
    // Kamera (Nacktmull-Pullover: doubles)
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

    // Zoom V3 Silk-Lite: persistenter Zustand auf Default
    zoomV3State = {};

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

    const size_t it_bytes       = totalPixels * sizeof(int);
    const size_t entropy_bytes  = size_t(numTiles) * sizeof(float);
    const size_t contrast_bytes = size_t(numTiles) * sizeof(float);

    // Progressive bytes (nur relevant, wenn Flag an)
    const bool progressiveOn = Settings::progressiveEnabled;
    const size_t z_bytes     = progressiveOn ? (totalPixels * sizeof(float2)) : 0;
    const size_t it_bytes2   = progressiveOn ? (totalPixels * sizeof(int))    : 0;

    // 🦦 Otter: Fast-Path – wenn alles schon passt, sofort raus (keine Logs)
    bool sizesOk =
        d_iterations.size() >= it_bytes      &&
        d_entropy.size()    >= entropy_bytes &&
        d_contrast.size()   >= contrast_bytes &&
        h_entropy.size()    == size_t(numTiles) &&
        h_contrast.size()   == size_t(numTiles) &&
        lastTileSize        == tileSize;

    if (progressiveOn) {
        sizesOk = sizesOk &&
                  d_stateZ.size()  >= z_bytes &&
                  d_stateIt.size() >= it_bytes2;
    }

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

    // --- grow decisions BEFORE allocation (für korrektes Zeroing) ---
    const size_t have_it        = d_iterations.size();
    const size_t have_entropy   = d_entropy.size();
    const size_t have_contrast  = d_contrast.size();
    const bool   grow_it        = have_it       < it_bytes;
    const bool   grow_entropy   = have_entropy  < entropy_bytes;
    const bool   grow_contrast  = have_contrast < contrast_bytes;

    // Progressive grow decisions
    const size_t have_z         = progressiveOn ? d_stateZ.size()  : 0;
    const size_t have_it2       = progressiveOn ? d_stateIt.size() : 0;
    const bool   grow_z         = progressiveOn && (have_z  < z_bytes);
    const bool   grow_it2       = progressiveOn && (have_it2 < it_bytes2);

    // --- iterations buffer (only-grow policy) ---
    if (grow_it) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_iterations grow: have=%zu -> need=%zu", have_it, it_bytes);
        }
        d_iterations.allocate(it_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_iterations ok: have=%zu need=%zu (no realloc)", have_it, it_bytes);
    }

    // --- entropy buffer (only-grow policy) ---
    if (grow_entropy) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_entropy grow: have=%zu -> need=%zu (tiles %d)", have_entropy, entropy_bytes, numTiles);
        }
        d_entropy.allocate(entropy_bytes);

        // Optional: Pointer-Attribute prüfen – nur wenn wirklich neu allokiert
        if constexpr (Settings::debugLogging) {
            cudaPointerAttributes attr{};
            const cudaError_t attrErr = cudaPointerGetAttributes(&attr, d_entropy.get());
            if (attrErr == cudaSuccess) {
                LUCHS_LOG_HOST("[CHECK] d_entropy attr: type=%d device=%d hostPtr=%p devicePtr=%p",
                               (int)attr.type, (int)attr.device,
                               (void*)attr.hostPointer, (void*)attr.devicePointer);
            } else {
                LUCHS_LOG_HOST("[CHECK] d_entropy attr query failed: err=%d", (int)attrErr);
            }
        }
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_entropy ok: have=%zu need=%zu (tiles %d, no realloc)", have_entropy, entropy_bytes, numTiles);
    }

    // --- contrast buffer (only-grow policy) ---
    if (grow_contrast) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_contrast grow: have=%zu -> need=%zu (tiles %d)", have_contrast, contrast_bytes, numTiles);
        }
        d_contrast.allocate(contrast_bytes);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ALLOC] d_contrast ok: have=%zu need=%zu (tiles %d, no realloc)", have_contrast, contrast_bytes, numTiles);
    }

    // 🐭 Maus: Zeroing nur bei NEU-Allocation (oder in Debug immer)
    const bool clearIterations = grow_it      || Settings::debugLogging;
    const bool clearEntropy    = grow_entropy || Settings::debugLogging;
    const bool clearContrast   = grow_contrast|| Settings::debugLogging;

    if (clearIterations) CUDA_CHECK(cudaMemset(d_iterations.get(), 0, it_bytes));
    if (clearEntropy)    CUDA_CHECK(cudaMemset(d_entropy.get(),    0, entropy_bytes));
    if (clearContrast)   CUDA_CHECK(cudaMemset(d_contrast.get(),   0, contrast_bytes));

    // -------------------- Progressive state buffers (Keks 4) --------------------
    if (progressiveOn) {
        if (grow_z) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_stateZ grow: have=%zu -> need=%zu (pixels %zu)", have_z, z_bytes, totalPixels);
            }
            d_stateZ.allocate(z_bytes);
            CUDA_CHECK(cudaMemset(d_stateZ.get(), 0, z_bytes));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_stateZ ok: have=%zu need=%zu (no realloc)", have_z, z_bytes);
        }

        if (grow_it2) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_stateIt grow: have=%zu -> need=%zu (pixels %zu)", have_it2, it_bytes2, totalPixels);
            }
            d_stateIt.allocate(it_bytes2);
            CUDA_CHECK(cudaMemset(d_stateIt.get(), 0, it_bytes2));
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_stateIt ok: have=%zu need=%zu (no realloc)", have_it2, it_bytes2);
        }

        if constexpr (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[MEM] progressive: stateZ=%zuB stateIt=%zuB totalPixels=%zu",
                           z_bytes, it_bytes2, totalPixels);
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
        LUCHS_LOG_HOST("[ALLOC] buffers ready: it=%p(%zu) entropy=%p(%zu) contrast=%p(%zu)%s | %dx%d px, tileSize=%d -> tiles=%d",
                       d_iterations.get(), d_iterations.size(),
                       d_entropy.get(),    d_entropy.size(),
                       d_contrast.get(),   d_contrast.size(),
                       progressiveOn ? " + progressive state" : "",
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

    // Progressive: beim Resize ebenfalls freigeben (sauberer Neustart der States)
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

    // Host-Timings fuer neues Frame auf Null
    lastTimings.resetHostFrame();

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated (progressive=%d)",
                       width, height, (int)Settings::progressiveEnabled);
    }
}
