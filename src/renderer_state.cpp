///// Otter: Einheitliche, klare Struktur – nur aktive Zustaende; Header schlank, keine PCH; Nacktmull-Pullover.
///// Schneefuchs: Speicher/Buffer exakt definiert; Host-Timings zentral – eine Quelle; /WX-fest; ASCII-only.
///// Maus: Progressive-Defaults aus Settings::progressiveEnabled; Cooldown/State robust.
///// Datei: src/renderer_state.cpp

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "renderer_resources.hpp"
#include <algorithm> // std::clamp, std::max

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
    
    progressiveEnabled         = Settings::progressiveEnabled;
    progressiveCooldownFrames  = 0;

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

    const size_t it_bytes       = totalPixels * sizeof(uint16_t);
    const size_t entropy_bytes  = size_t(numTiles) * sizeof(float);
    const size_t contrast_bytes = size_t(numTiles) * sizeof(float);

    // Progressive-States (nur wenn Feature global aktiviert)
    const bool   wantProg       = Settings::progressiveEnabled;
    const size_t z_bytes        = totalPixels * sizeof(float2);
    const size_t it2_bytes      = totalPixels * sizeof(uint16_t);

    // 🦦 Otter: Fast-Path – wenn alles schon passt, sofort raus
    const bool sizesOk =
        d_iterations.size() >= it_bytes      &&
        d_entropy.size()    >= entropy_bytes &&
        d_contrast.size()   >= contrast_bytes &&
        (!wantProg || (d_stateZ.size() >= z_bytes && d_stateIt.size() >= it2_bytes)) &&
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
        const bool   grow = have < it_bytes;
        if (grow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_iterations grow: have=%zu -> need=%zu", have, it_bytes);
            }
            d_iterations.allocate(it_bytes);
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_iterations ok: have=%zu need=%zu (no realloc)", have, it_bytes);
        }
        if (grow || Settings::debugLogging) CUDA_CHECK(cudaMemset(d_iterations.get(), 0, it_bytes));
    }

    // --- entropy buffer (only-grow policy) ---
    {
        const size_t have = d_entropy.size();
        const bool   grow = have < entropy_bytes;
        if (grow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_entropy grow: have=%zu -> need=%zu (tiles %d)", have, entropy_bytes, numTiles);
            }
            d_entropy.allocate(entropy_bytes);
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_entropy ok: have=%zu need=%zu (tiles %d, no realloc)", have, entropy_bytes, numTiles);
        }
        if (grow || Settings::debugLogging) CUDA_CHECK(cudaMemset(d_entropy.get(), 0, entropy_bytes));
    }

    // Optional: Pointer-Attribute prüfen
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

    // --- contrast buffer (only-grow policy) ---
    {
        const size_t have = d_contrast.size();
        const bool   grow = have < contrast_bytes;
        if (grow) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_contrast grow: have=%zu -> need=%zu (tiles %d)", have, contrast_bytes, numTiles);
            }
            d_contrast.allocate(contrast_bytes);
        } else if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ALLOC] d_contrast ok: have=%zu need=%zu (tiles %d, no realloc)", have, contrast_bytes, numTiles);
        }
        if (grow || Settings::debugLogging) CUDA_CHECK(cudaMemset(d_contrast.get(), 0, contrast_bytes));
    }

    // --- progressive buffers (only-grow policy; nur wenn Feature aktiv) ---
    if (wantProg) {
        // z-state
        {
            const size_t have = d_stateZ.size();
            const bool   grow = have < z_bytes;
            if (grow) {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ALLOC] d_stateZ grow: have=%zu -> need=%zu (px %zu)", have, z_bytes, totalPixels);
                }
                d_stateZ.allocate(z_bytes);
            } else if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_stateZ ok: have=%zu need=%zu (no realloc)", have, z_bytes);
            }
            if (grow || Settings::debugLogging) CUDA_CHECK(cudaMemset(d_stateZ.get(), 0, z_bytes));
        }
        // it-state
        {
            const size_t have = d_stateIt.size();
            const bool   grow = have < it2_bytes;
            if (grow) {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ALLOC] d_stateIt grow: have=%zu -> need=%zu (px %zu)", have, it2_bytes, totalPixels);
                }
                d_stateIt.allocate(it2_bytes);
            } else if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_stateIt ok: have=%zu need=%zu (no realloc)", have, it2_bytes);
            }
            if (grow || Settings::debugLogging) CUDA_CHECK(cudaMemset(d_stateIt.get(), 0, it2_bytes));
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
        LUCHS_LOG_HOST("[ALLOC] buffers ready: it=%p(%zu) entropy=%p(%zu) contrast=%p(%zu) z=%p(%zu) it2=%p(%zu) | %dx%d px, tileSize=%d -> tiles=%d",
                       d_iterations.get(), d_iterations.size(),
                       d_entropy.get(),    d_entropy.size(),
                       d_contrast.get(),   d_contrast.size(),
                       wantProg ? d_stateZ.get()  : nullptr, wantProg ? d_stateZ.size()  : 0,
                       wantProg ? d_stateIt.get() : nullptr, wantProg ? d_stateIt.size() : 0,
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
    d_stateZ.free();
    d_stateIt.free();

    // Unregister CUDA-GL PBO
    CudaInterop::unregisterAllPBOs();

    // Free GL buffers via RAII
    for (auto& b : pboRing) { b.free(); }
    tex.free();

    // Apply new size
    width  = newWidth;
    height = newHeight;

    OpenGLUtils::setGLResourceContext("resize");

    // Create fresh GL buffers
    for (auto& b : pboRing) { b = Hermelin::GLBuffer(OpenGLUtils::createPBO(width, height)); }
    pboIndex = 0;
    tex = Hermelin::GLBuffer(OpenGLUtils::createTexture(width, height));

    CudaInterop::registerPBO(currentPBO());

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
        LUCHS_LOG_HOST("[RESIZE] %d x %d buffers reallocated", width, height);
    }
}

// 🧯 Progressive-State vorsichtig invalidieren (1-Frame-Cooldown, ohne Throw)
void RendererState::invalidateProgressiveState(bool hardReset) noexcept {
    // Soft: nur Pause des Resume-Mechanismus, keine Memsets hier
    progressiveEnabled        = false;
    // Mini-Cooldown: 2 Frames Puffer, damit Kernel/Upload stabil sind
    progressiveCooldownFrames = 2;

    if (hardReset) {
        // Harte Invalidierung optional: Inhalte werden bei nächster setupCudaBuffers()
        // via cudaMemset initialisiert; hier keine CUDA-Calls (noexcept!).
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PROG] hardReset requested (state will be cleared on next allocation)");
        }
    } else {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PROG] soft invalidate: cooldown=%d", progressiveCooldownFrames);
        }
    }
}
