///// Otter: RAII for CUDA/GL buffers; deterministic lifecycle; ASCII-only API docs.
///// Schneefuchs: Clear ownership; header/source in sync; no hidden allocations.
///// Maus: Minimal surface area; no manual cudaFree/glDelete in client code.

#pragma once

#include <GL/glew.h>
#include <cstddef>

namespace Hermelin {

class CudaDeviceBuffer {
public:
    CudaDeviceBuffer();
    ~CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    // Allocate exactly sizeBytes (previous content is discarded).
    void   allocate(size_t sizeBytes);
    // Alias for allocate for clients expecting a resize() API.
    void   resize(size_t sizeBytes);
    // Ensure capacity of at least minBytes; never shrinks.
    void   ensure(size_t minBytes);

    void   free();

    void*  get() const;
    size_t size() const;
    [[nodiscard]] explicit operator bool() const { return get() != nullptr && size() > 0; }

private:
    void*  ptr_;
    size_t sizeBytes_;
};

class GLBuffer {
public:
    GLBuffer();
    ~GLBuffer();

    GLBuffer(const GLBuffer&) = delete;
    GLBuffer& operator=(const GLBuffer&) = delete;

    GLBuffer(GLBuffer&& other) noexcept;
    GLBuffer& operator=(GLBuffer&& other) noexcept;

    explicit GLBuffer(GLuint id) noexcept;

    void   create();
    void   allocate(GLsizeiptr sizeBytes, GLenum usage = GL_DYNAMIC_DRAW);
    // Initialize as pixel-unpack buffer for RGBA8 (bytesPerPixel default = 4).
    void   initAsPixelBuffer(int width, int height, int bytesPerPixel = 4);
    void   free();

    [[nodiscard]] GLuint id() const;
    [[nodiscard]] explicit operator bool() const { return id() != 0; }

private:
    GLuint id_;
};

} // namespace Hermelin
