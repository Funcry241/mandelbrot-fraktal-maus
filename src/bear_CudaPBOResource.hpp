#pragma once
#ifndef BEAR_CUDA_PBO_RESOURCE_HPP
#define BEAR_CUDA_PBO_RESOURCE_HPP

#include <cstddef>          // size_t
#include <GL/glew.h>        // GLuint
#include <cuda_gl_interop.h>
// Avoid leaking CUDA vector types in public headers.
struct uchar4; // forward-declare for pointer-only usage

namespace CudaInterop {

// Verwaltet Lifetime & Mapping eines CUDA-GL PBO-Interop-Handles
class bear_CudaPBOResource {
public:
    explicit bear_CudaPBOResource(GLuint pboId);
    ~bear_CudaPBOResource();

    // Nicht kopierbar; beweglich
    bear_CudaPBOResource(const bear_CudaPBOResource&) = delete;
    bear_CudaPBOResource& operator=(const bear_CudaPBOResource&) = delete;
    bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept;
    bear_CudaPBOResource& operator=(bear_CudaPBOResource&& other) noexcept;

    // Handle & Status
    [[nodiscard]] cudaGraphicsResource_t get() const noexcept;
    [[nodiscard]] bool   isMapped() const noexcept;
    [[nodiscard]] size_t lastSize() const noexcept;

    // Mapping-API (kompatibel) + Guard-Overload
    [[nodiscard]] void*   mapAndLog(size_t& sizeOut);
    [[nodiscard]] uchar4* mapAndLog();
    [[nodiscard]] uchar4* mapAndLogExpect(size_t expectedBytes); // neu: Größe prüfen

    // Unmapping
    void unmap();
    void unmapAndLog();

private:
    cudaGraphicsResource_t resource_{nullptr};
    bool   mapped_{false};
    size_t lastSize_{0};
};

} // namespace CudaInterop

#endif // BEAR_CUDA_PBO_RESOURCE_HPP
