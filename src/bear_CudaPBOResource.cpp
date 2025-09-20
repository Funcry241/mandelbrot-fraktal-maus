///// Otter: OpenGL PBO interop; map/unmap + pointer retrieval logged deterministically.
///// Schneefuchs: No shared state here; pure RAII around cudaGraphicsResource.
///// Maus: Numeric CUDA rc only via CUDA_CHECK; ASCII logs; quiet when disabled.
///// Datei: src/bear_CudaPBOResource.cpp
#include "pch.hpp"
#include "bear_CudaPBOResource.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace CudaInterop {

bear_CudaPBOResource::bear_CudaPBOResource(unsigned int glBufferId)
    : m_glId(glBufferId), m_gr(nullptr)
{
    cudaError_t rc = cudaGraphicsGLRegisterBuffer(&m_gr, m_glId, cudaGraphicsRegisterFlagsNone);
    if (rc != cudaSuccess) {
        m_gr = nullptr;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CUDA-Interop][PBO] register id=%u failed rc=%d", m_glId, (int)rc);
        }
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CUDA-Interop][PBO] registered id=%u", m_glId);
    }
}

bear_CudaPBOResource::~bear_CudaPBOResource() {
    if (m_gr) {
        cudaError_t rc = cudaGraphicsUnregisterResource(m_gr);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CUDA-Interop][PBO] unregister id=%u rc=%d", m_glId, (int)rc);
        }
        (void)rc;
        m_gr = nullptr;
    }
}

cudaGraphicsResource* bear_CudaPBOResource::get() const {
    return m_gr;
}

void* bear_CudaPBOResource::mapAndLog(size_t& bytesOut) {
    bytesOut = 0;
    if (!m_gr) return nullptr;

    CUDA_CHECK(cudaGraphicsMapResources(1, &m_gr, 0));

    void*  ptr   = nullptr;
    size_t bytes = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &bytes, m_gr));
    bytesOut = bytes;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][MAP] id=%u ptr=%p bytes=%zu", m_glId, ptr, bytes);
    }
    return ptr;
}

void bear_CudaPBOResource::unmap() {
    if (!m_gr) return;
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_gr, 0));
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PBO][UNMAP] id=%u", m_glId);
    }
}

} // namespace CudaInterop
