///// Otter: PBO-RAII – sicher unmap im Dtor; No-arg map & unmapAndLog.
///// Schneefuchs: deterministische ASCII-Logs; numerische CUDA-Codes.
///// Maus: Keine Side-Effects – GL-Bindings werden sauber restauriert.
///// Datei: src/bear_CudaPBOResource.cpp

#include "pch.hpp"
#include "bear_CudaPBOResource.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#include <chrono>
#include <utility>           // std::exchange
#include <GL/glew.h>
#include <cuda_gl_interop.h> // CUDA-GL interop API

namespace CudaInterop {

// 🐻 Konstruktor – registriert PBO bei Erstellung
bear_CudaPBOResource::bear_CudaPBOResource(GLuint pboId) {
    resource_ = nullptr;
    mapped_   = false;
    lastSize_ = 0;

    // 🦊 Preserve-then-bind; alten Binding-Status danach wiederherstellen.
    GLint prevBinding = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevBinding);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] Bind PBO %u (prev=%d)", pboId, prevBinding);
    }

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&resource_, pboId, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] cudaGraphicsGLRegisterBuffer code=%d", (int)err);
        }
        resource_ = nullptr;
    } else {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[DEBUG] Registered PBO %u as CUDA resource %p", pboId, (void*)resource_);
        }
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, (GLuint)prevBinding);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] Restore GL_PIXEL_UNPACK_BUFFER_BINDING to %d", prevBinding);
    }
}

// 🐻 Destruktor – unmap (falls nötig) und deregistrieren
bear_CudaPBOResource::~bear_CudaPBOResource() {
    if (resource_) {
        if (mapped_) {
            (void)cudaGraphicsUnmapResources(1, &resource_, 0);
            mapped_ = false;
        }
        cudaError_t err = cudaGraphicsUnregisterResource(resource_);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PBO] cudaGraphicsUnregisterResource code=%d", (int)err);
        }
        resource_ = nullptr;
        lastSize_ = 0;
    }
}

// 🐻 mapAndLog – mappt (idempotent), liefert DevPtr + Size (geloggt)
void* bear_CudaPBOResource::mapAndLog(size_t& sizeOut) {
    void* devPtr = nullptr;
    sizeOut = 0;

    if (!resource_) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] mapAndLog() called with null resource");
        }
        return nullptr;
    }

    cudaError_t err = cudaSuccess;
    auto tMapStart = std::chrono::high_resolution_clock::now();

    if (!mapped_) {
        err = cudaGraphicsMapResources(1, &resource_, 0);
        if constexpr (Settings::debugLogging) {
            auto tMapEnd = std::chrono::high_resolution_clock::now();
            const double mapMs = std::chrono::duration<double, std::milli>(tMapEnd - tMapStart).count();
            LUCHS_LOG_HOST("[PERF] MapResources: %.3f ms", mapMs);
            LUCHS_LOG_HOST("[PBO] cudaGraphicsMapResources code=%d", (int)err);
        }
        if (err != cudaSuccess) return nullptr;
        mapped_ = true;
    }

    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &sizeOut, resource_);
    lastSize_ = (err == cudaSuccess) ? sizeOut : 0;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] ResourceGetMappedPointer code=%d ptr=%p size=%zu", (int)err, devPtr, sizeOut);
    }
    if (err != cudaSuccess) {
        // Cleanup auf Fehlerfall, um geleakten Map-Status zu vermeiden
        (void)cudaGraphicsUnmapResources(1, &resource_, 0);
        mapped_ = false;
        devPtr  = nullptr;
        sizeOut = 0;
        lastSize_ = 0;
        return nullptr;
    }

    return devPtr;
}

// 🐻 Overload: typisierter Pixelpointer (uchar4), Groesse wird intern geloggt
uchar4* bear_CudaPBOResource::mapAndLog() {
    size_t sz = 0;
    void* p = mapAndLog(sz);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] mapAndLog(void) size=%zu bytes", sz);
    }
    return reinterpret_cast<uchar4*>(p);
}

// 🐻 Overload mit Größenprüfung (Guard gegen PBO-Mismatch)
uchar4* bear_CudaPBOResource::mapAndLogExpect(size_t expectedBytes) {
    size_t sz = 0;
    uchar4* p = mapAndLog();
    sz = lastSize_;
    if (p && expectedBytes && sz < expectedBytes) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] PBO mapped size too small: have=%zu need=%zu", sz, expectedBytes);
        }
        unmap(); // sauber zurücksetzen
        return nullptr;
    }
    return p;
}

// 🐻 unmap – nur wenn gemappt
void bear_CudaPBOResource::unmap() {
    if (!resource_ || !mapped_) return;
    const cudaError_t err = cudaGraphicsUnmapResources(1, &resource_, 0);
    mapped_ = false;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO] cudaGraphicsUnmapResources code=%d", (int)err);
    }
}

// 🐻 unmapAndLog – symmetrische Zeitmessung
void bear_CudaPBOResource::unmapAndLog() {
    if (!resource_ || !mapped_) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    unmap();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PERF] UnmapResources: %.3f ms", ms);
    }
}

// Getter
cudaGraphicsResource_t bear_CudaPBOResource::get() const noexcept { return resource_; }
bool  bear_CudaPBOResource::isMapped() const noexcept { return mapped_; }
size_t bear_CudaPBOResource::lastSize() const noexcept { return lastSize_; }

// Moves
bear_CudaPBOResource::bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept
: resource_(std::exchange(other.resource_, nullptr)),
  mapped_(std::exchange(other.mapped_, false)),
  lastSize_(std::exchange(other.lastSize_, 0)) {}

bear_CudaPBOResource& bear_CudaPBOResource::operator=(bear_CudaPBOResource&& other) noexcept {
    if (this != &other) {
        if (resource_) {
            if (mapped_) (void)cudaGraphicsUnmapResources(1, &resource_, 0);
            (void)cudaGraphicsUnregisterResource(resource_);
        }
        resource_ = std::exchange(other.resource_, nullptr);
        mapped_   = std::exchange(other.mapped_, false);
        lastSize_ = std::exchange(other.lastSize_, 0);
    }
    return *this;
}

} // namespace CudaInterop
