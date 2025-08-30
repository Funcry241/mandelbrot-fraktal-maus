///// Otter: PBO-RAII – sicher unmap im Dtor; No-arg map & unmapAndLog.
///// Schneefuchs: deterministische ASCII-Logs; numerische CUDA-Codes.
///// Maus: Keine Side-Effects – GL-Bindings werden sauber restauriert.
#include "bear_CudaPBOResource.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#include <chrono>
#include <utility>           // std::exchange
#include <GL/glew.h>
#include <cuda_gl_interop.h> // CUDA-GL interop API

namespace CudaInterop {

// 🐻 Bär: Konstruktor – registriert PBO bei Erstellung
bear_CudaPBOResource::bear_CudaPBOResource(GLuint pboId) {
    resource_ = nullptr;

    // 🦊 Schneefuchs: Preserve-then-bind; alten Binding-Status danach wiederherstellen.
    GLint prevBinding = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevBinding);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] Bind PBO %u (prev=%d)", pboId, prevBinding);
    }

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&resource_, pboId, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] cudaGraphicsGLRegisterBuffer code=%d", static_cast<int>(err));
        }
        resource_ = nullptr;
    } else {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[DEBUG] Registered PBO %u as CUDA resource %p", pboId, reinterpret_cast<void*>(resource_));
        }
    }

    // 🦊 Schneefuchs: Restore previous binding to avoid side effects.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevBinding));
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] Restore GL_PIXEL_UNPACK_BUFFER_BINDING to %d", prevBinding);
    }
}

// 🐻 Bär: Destruktor – deregistriert CUDA-Resource (robust: vorher unmap versuchen)
bear_CudaPBOResource::~bear_CudaPBOResource() {
    if (resource_) {
        // Best effort: Unmap, falls noch gemappt (fehlerrobust ignorieren)
        (void)cudaGraphicsUnmapResources(1, &resource_, 0);
        cudaError_t err = cudaGraphicsUnregisterResource(resource_);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PBO] cudaGraphicsUnregisterResource code=%d", static_cast<int>(err));
        }
        resource_ = nullptr;
    }
}

// 🐻 Bär: mapAndLog – mappt und loggt DevPtr + Size
void* bear_CudaPBOResource::mapAndLog(size_t& sizeOut) {
    void* devPtr = nullptr;
    sizeOut = 0;

    if (!resource_) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] mapAndLog() called with null resource");
        }
        return nullptr;
    }

    // 🦊 Schneefuchs: Messung vor Map-Call zur Blockanalyse.
    auto tMapStart = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGraphicsMapResources(1, &resource_, 0);
    auto tMapEnd = std::chrono::high_resolution_clock::now();

    const double mapMs = std::chrono::duration<double, std::milli>(tMapEnd - tMapStart).count();
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PERF] MapResources: %.3f ms", mapMs);
        LUCHS_LOG_HOST("[PBO] cudaGraphicsMapResources code=%d", static_cast<int>(err));
    }
    if (err != cudaSuccess) return nullptr;

    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &sizeOut, resource_);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] ResourceGetMappedPointer code=%d ptr=%p size=%zu", static_cast<int>(err), devPtr, sizeOut);
    }
    if (err != cudaSuccess) {
        // 🦦 Otter: Cleanup on failure to avoid leaked map.
        const cudaError_t unmapErr = cudaGraphicsUnmapResources(1, &resource_, 0);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PBO] Unmap after GetMappedPointer failure code=%d", static_cast<int>(unmapErr));
        }
        devPtr = nullptr;
        sizeOut = 0;
        return nullptr;
    }

    return devPtr;
}

// 🐻 Bär: Bequemer Overload – typisiert
uchar4* bear_CudaPBOResource::mapAndLog() {
    size_t sz = 0;
    void* p = mapAndLog(sz);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] mapAndLog(void) size=%zu bytes", sz);
    }
    return reinterpret_cast<uchar4*>(p);
}

// 🐻 Bär: unmap – gibt gemappte Resource frei
void bear_CudaPBOResource::unmap() {
    if (!resource_) return;
    const cudaError_t err = cudaGraphicsUnmapResources(1, &resource_, 0);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] cudaGraphicsUnmapResources code=%d", static_cast<int>(err));
    }
}

// 🐻 Bär: unmapAndLog – mit Zeitmessung (symmetrisch zu Map-Perf)
void bear_CudaPBOResource::unmapAndLog() {
    if (!resource_) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    unmap();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PERF] UnmapResources: %.3f ms", ms);
    }
}

// 🐻 Bär: Getter für das Resource-Handle
cudaGraphicsResource_t bear_CudaPBOResource::get() const noexcept {
    return resource_;
}

// 🐻 Bär: Move-Konstruktor – übernimmt Ownership
bear_CudaPBOResource::bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept
: resource_(std::exchange(other.resource_, nullptr)) {}

// 🐻 Bär: Move-Assignment – übernimmt Ownership und räumt auf
bear_CudaPBOResource& bear_CudaPBOResource::operator=(bear_CudaPBOResource&& other) noexcept {
    if (this != &other) {
        if (resource_) {
            (void)cudaGraphicsUnregisterResource(resource_);
        }
        resource_ = std::exchange(other.resource_, nullptr);
    }
    return *this;
}

} // namespace CudaInterop
