// Datei: src/hermelin_buffer.hpp
// 🐜 Hermelin: RAII-Wrapper für CUDA- und OpenGL-Buffer mit deterministic Lifecycle-Management.
// 🦦 Otter: Sichere Ressourcenverwaltung, automatische Freigabe beim Destruktor.
// 🦊 Schneefuchs: Klarer Ownership-Begriff, keine manuellen cudaFree/glDeleteCalls im Client-Code.

#pragma once

#include <cuda_runtime.h>
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

    void allocate(size_t sizeBytes);
    void free();

    void* get() const;
    size_t size() const;
    explicit operator bool() const;

private:
    void* ptr_;
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

    void create();
    void allocate(GLsizeiptr sizeBytes, GLenum usage = GL_DYNAMIC_DRAW);
    void free();

    GLuint id() const;
    explicit operator bool() const;

private:
    GLuint id_;
};

} // namespace Hermelin
