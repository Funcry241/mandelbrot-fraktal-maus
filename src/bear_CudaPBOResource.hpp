///// Otter: RAII wrapper for CUDA-GL PBO interop.
///// Schneefuchs: Header stays light; no state/gl headers here.
///// Maus: Deterministic; ASCII logs when enabled.
///// Datei: src/bear_CudaPBOResource.hpp
#pragma once
#include <cstddef> // size_t

// Forward declare to keep header light
struct cudaGraphicsResource;

namespace CudaInterop {

class bear_CudaPBOResource {
public:
    explicit bear_CudaPBOResource(unsigned int glBufferId);
    ~bear_CudaPBOResource();

    cudaGraphicsResource* get() const;
    void*  mapAndLog(size_t& bytesOut);
    void   unmap();

    bear_CudaPBOResource(const bear_CudaPBOResource&) = delete;
    bear_CudaPBOResource& operator=(const bear_CudaPBOResource&) = delete;

private:
    unsigned int          m_glId = 0;
    cudaGraphicsResource* m_gr   = nullptr;
};

} // namespace CudaInterop
