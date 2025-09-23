///// Otter: Streams mit Prioritäten – Render hoch, Copy niedrig; keine GL-Abhängigkeit; ASCII-Logs.
///// Schneefuchs: Fallback sicher (Range optional); /WX-fest; API unverändert; keine Host-Syncs außer Debug.
///// Maus: Klare Logs [STREAM]; deterministisches Verhalten; unter 300 Zeilen.
///// Datei: src/renderer_state_cuda.cpp

#include "pch.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "common.hpp"
#include "luchs_log_host.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace {
// ----- Tiles (CUDA-seitig genutzt) --------------------------------------------
inline void computeTiles(int width, int height, int tileSize,
                         int& tilesX, int& tilesY, int& numTiles) noexcept {
    tilesX   = (width  + tileSize - 1) / tileSize;
    tilesY   = (height + tileSize - 1) / tileSize;
    numTiles = tilesX * tilesY;
}
} // namespace

// =============================== Streams / Events =============================

void RendererState::createCudaStreamsIfNeeded() {
    CUDA_CHECK(cudaSetDevice(0));

    // Priority-Range abfragen (lo = niedrigste Priorität, hi = höchste; numerisch: hi <= lo)
    int lo = 0, hi = 0;
    const cudaError_t prc = cudaDeviceGetStreamPriorityRange(&lo, &hi);
    const bool hasRange = (prc == cudaSuccess);
    const int prRender = hasRange ? hi : 0; // höchste verfügbare Priorität
    const int prCopy   = hasRange ? lo : 0; // niedrigste verfügbare Priorität

    if (!renderStream) {
        cudaError_t e = cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, prRender);
        if (e != cudaSuccess) {
            // Fallback ohne Priorität
            renderStream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&renderStream, cudaStreamNonBlocking));
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[STREAM] renderStream created %p (non-blocking, prio=default, range-avail=%d)",
                               (void*)renderStream, hasRange ? 1 : 0);
            }
        } else {
            if constexpr (Settings::debugLogging) {
                if (hasRange) {
                    LUCHS_LOG_HOST("[STREAM] renderStream created %p (non-blocking, prio=%d, range hi=%d lo=%d)",
                                   (void*)renderStream, prRender, hi, lo);
                } else {
                    LUCHS_LOG_HOST("[STREAM] renderStream created %p (non-blocking, prio=default, no-range)",
                                   (void*)renderStream);
                }
            }
        }
    }

    if (!copyStream) {
        cudaError_t e = cudaStreamCreateWithPriority(&copyStream, cudaStreamNonBlocking, prCopy);
        if (e != cudaSuccess) {
            // Fallback ohne Priorität
            copyStream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[STREAM] copyStream created %p (non-blocking, prio=default, range-avail=%d)",
                               (void*)copyStream, hasRange ? 1 : 0);
            }
        } else {
            if constexpr (Settings::debugLogging) {
                if (hasRange) {
                    LUCHS_LOG_HOST("[STREAM] copyStream created %p (non-blocking, prio=%d, range hi=%d lo=%d)",
                                   (void*)copyStream, prCopy, hi, lo);
                } else {
                    LUCHS_LOG_HOST("[STREAM] copyStream created %p (non-blocking, prio=default, no-range)",
                                   (void*)copyStream);
                }
            }
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

// ================================== Ctor (kein Dtor hier!) ====================

RendererState::RendererState(int w, int h)
: width(w), height(h) {
    // Streams/Events first, Reset handled in GL-TU (calls back into CUDA helpers)
    createCudaStreamsIfNeeded();
    createCudaEventsIfNeeded();
    // Reset ist in renderer_state_gl.cpp implementiert (benötigt GL-Fence-Clear)
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
        const size_t haveIt = d_iterations.size();
        if (haveIt < it_bytes) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_iterations grow: %zu -> %zu", haveIt, it_bytes);
            }
            d_iterations.allocate(it_bytes);
        }

        const size_t haveE = d_entropy.size();
        if (haveE < entropy_bytes) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_entropy grow: %zu -> %zu (tiles %d)", haveE, entropy_bytes, numTiles);
            }
            d_entropy.allocate(entropy_bytes);
        }

        const size_t haveC = d_contrast.size();
        if (haveC < contrast_bytes) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ALLOC] d_contrast grow: %zu -> %zu (tiles %d)", haveC, contrast_bytes, numTiles);
            }
            d_contrast.allocate(contrast_bytes);
        }

        if (wantProg) {
            const size_t haveZ = d_stateZ.size();
            if (haveZ < z_bytes) {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ALLOC] d_stateZ grow: %zu -> %zu (px %zu)", haveZ, z_bytes, totalPixels);
                }
                d_stateZ.allocate(z_bytes);
            }
            const size_t haveIt2 = d_stateIt.size();
            if (haveIt2 < it2_bytes) {
                if constexpr (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[ALLOC] d_stateIt grow: %zu -> %zu (px %zu)", haveIt2, it2_bytes, totalPixels);
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

    // Host mirrors (reserve worst case by MIN_TILE_SIZE, then size exact)
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
