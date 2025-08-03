// Datei: src/bear_CudaPBOResource.cpp
// 🐻 Bär: Implementation des RAII-Wrapper für CUDA-PBO.
// 🐻 Bär: Teil des Projekts "Bär" für robustes Ressourcenmanagement.

#include "bear_CudaPBOResource.hpp"
#include "luchs_log_host.hpp"
#include <chrono>
#include <GL/glew.h>

namespace CudaInterop {

// 🐻 Bär: Konstruktor – registriert PBO bei Erstellung
bear_CudaPBOResource::bear_CudaPBOResource(GLuint pboId) {
    // 🐻 Otter: explizit sicherstellen, dass GL-Buffer gebunden ist
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
    GLint bound = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &bound);
    LUCHS_LOG_HOST("[PBO] Bound PBO %u, GL_PIXEL_UNPACK_BUFFER_BINDING = %d", pboId, bound);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&resource_, pboId, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        LUCHS_LOG_HOST("[ERROR] cudaGraphicsGLRegisterBuffer failed: %s", cudaGetErrorString(err));
        resource_ = nullptr;
    } else {
        LUCHS_LOG_HOST("[DEBUG] Registered PBO %u as CUDA resource %p", pboId, (void*)resource_);
    }
}

// 🐻 Bär: Destruktor – deregistriert CUDA-Resource bei Zerstörung
bear_CudaPBOResource::~bear_CudaPBOResource() {
    if (resource_) {
        cudaError_t err = cudaGraphicsUnregisterResource(resource_);
        if (err != cudaSuccess) {
            LUCHS_LOG_HOST("[ERROR] cudaGraphicsUnregisterResource failed: %s", cudaGetErrorString(err));
        } else {
            LUCHS_LOG_HOST("[DEBUG] Unregistered CUDA resource %p", (void*)resource_);
        }
    }
}

// 🐻 Bär: mapAndLog – mappt und loggt DevPtr + Size
void* bear_CudaPBOResource::mapAndLog(size_t& sizeOut) {
    void* devPtr = nullptr;
    sizeOut = 0;

    if (!resource_) {
        LUCHS_LOG_HOST("[ERROR] mapAndLog() called with null resource.");
        return nullptr;
    }

    // 🐑 Schneefuchs: Messung vor Map-Call zur Blockanalyse
    auto tMapStart = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGraphicsMapResources(1, &resource_);
    auto tMapEnd = std::chrono::high_resolution_clock::now();

    double mapMs = std::chrono::duration<double, std::milli>(tMapEnd - tMapStart).count();
    LUCHS_LOG_HOST("[PERF] MapResources: %.3f ms", mapMs);

    LUCHS_LOG_HOST("[PBO] cudaGraphicsMapResources returned %d", static_cast<int>(err));
    if (err != cudaSuccess) return nullptr;

    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &sizeOut, resource_);
    LUCHS_LOG_HOST("[PBO] Mapped pointer = %p, size = %zu, err = %d", devPtr, sizeOut, static_cast<int>(err));

    return devPtr;
}

// 🐻 Bär: unmap – gibt gemappte Resource frei
void bear_CudaPBOResource::unmap() {
    if (!resource_) return;
    cudaError_t err = cudaGraphicsUnmapResources(1, &resource_);
    LUCHS_LOG_HOST("[PBO] cudaGraphicsUnmapResources returned %d", static_cast<int>(err));
}

// 🐻 Bär: Getter für das Resource-Handle
cudaGraphicsResource_t bear_CudaPBOResource::get() const noexcept {
    return resource_;
}

// 🐻 Bär: Move-Konstruktor – übernimmt Ownership
bear_CudaPBOResource::bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept
: resource_(other.resource_) {
    other.resource_ = nullptr;
}

// 🐻 Bär: Move-Assignment – übernimmt Ownership und räumt auf
bear_CudaPBOResource& bear_CudaPBOResource::operator=(bear_CudaPBOResource&& other) noexcept {
    if (this != &other) {
        if (resource_) {
            cudaGraphicsUnregisterResource(resource_);
        }
        resource_ = other.resource_;
        other.resource_ = nullptr;
    }
    return *this;
}

} // namespace CudaInterop
