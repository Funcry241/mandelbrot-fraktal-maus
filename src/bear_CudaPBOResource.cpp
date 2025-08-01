// Datei: src/bear_CudaPBOResource.cpp
// 🐻 Bär: Implementation des RAII-Wrapper für CUDA-PBO.
// 🐻 Bär: Teil des Projekts "Bär" für robustes Ressourcenmanagement.

#include "bear_CudaPBOResource.hpp"
#include "luchs_log_host.hpp"

namespace CudaInterop {

// 🐻 Bär: Konstruktor – registriert PBO bei Erstellung
bear_CudaPBOResource::bear_CudaPBOResource(GLuint pboId) {
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&resource_, pboId, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        LUCHS_LOG_HOST("[ERROR] Bär: cudaGraphicsGLRegisterBuffer failed: %s", cudaGetErrorString(err));
        resource_ = nullptr;
    } else {
        LUCHS_LOG_HOST("[DEBUG] Bär: Registered PBO %u as CUDA resource %p", pboId, (void*)resource_);
    }
}

// 🐻 Bär: Destruktor – deregistriert CUDA-Resource bei Zerstörung
bear_CudaPBOResource::~bear_CudaPBOResource() {
    if (resource_) {
        cudaError_t err = cudaGraphicsUnregisterResource(resource_);
        if (err != cudaSuccess) {
            LUCHS_LOG_HOST("[ERROR] Bär: cudaGraphicsUnregisterResource failed: %s", cudaGetErrorString(err));
        } else {
            LUCHS_LOG_HOST("[DEBUG] Bär: Unregistered CUDA resource %p", (void*)resource_);
        }
    }
}

// 🐻 Bär: Getter für das Resource-Handle
cudaGraphicsResource_t bear_CudaPBOResource::get() const noexcept {
    return resource_;
}

// 🐻 Bär: Move-Konstruktor – übernimmt Ownership
bear_CudaPBOResource::bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept
: resource_(other.resource_) {
    other.resource_ = nullptr;  // Bär: clear source
}

// 🐻 Bär: Move-Assignment – übernimmt Ownership und räumt auf
bear_CudaPBOResource& bear_CudaPBOResource::operator=(bear_CudaPBOResource&& other) noexcept {
    if (this != &other) {
        if (resource_) {
            cudaGraphicsUnregisterResource(resource_);  // Bär: clean existing
        }
        resource_ = other.resource_;
        other.resource_ = nullptr;
    }
    return *this;
}

} // namespace CudaInterop
