// Datei: src/hermelin_buffer.hpp
// 🐜 Hermelin: RAII-Wrapper für CUDA- und OpenGL-Buffer mit deterministic Lifecycle-Management.
// 🦦 Otter: Sichere Ressourcenverwaltung, automatische Freigabe beim Destruktor.
// 🦊 Schneefuchs: Klarer Ownership-Begriff, keine manuellen cudaFree/glDeleteCalls im Client-Code.

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <GL/glew.h>

namespace Hermelin {

// RAII für CUDA Device Memory (z.B. d_iterations, d_entropy)
class CudaDeviceBuffer {
public:
CudaDeviceBuffer() : ptr_(nullptr), sizeBytes_(0) {}
~CudaDeviceBuffer() { free(); }

// Nicht kopierbar
CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

// Verschiebbar
CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : ptr_(other.ptr_), sizeBytes_(other.sizeBytes_) {
    other.ptr_ = nullptr;
    other.sizeBytes_ = 0;
}
CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        sizeBytes_ = other.sizeBytes_;
        other.ptr_ = nullptr;
        other.sizeBytes_ = 0;
    }
    return *this;
}

void allocate(size_t sizeBytes) {
    free();
    cudaError_t err = cudaMalloc(&ptr_, sizeBytes);
    if (err != cudaSuccess) {
        ptr_ = nullptr;
        sizeBytes_ = 0;
        throw std::runtime_error("Hermelin: cudaMalloc failed");
    }
    sizeBytes_ = sizeBytes;
}

void free() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        sizeBytes_ = 0;
    }
}

void* get() const { return ptr_; }
size_t size() const { return sizeBytes_; }

// Explizites Nullptr-Check für Sicherheit
explicit operator bool() const { return ptr_ != nullptr; }

private:
void* ptr_;
size_t sizeBytes_;
};

// RAII für OpenGL Buffer (z.B. PBO)
class GLBuffer {
public:
GLBuffer() : id_(0) {}
~GLBuffer() { free(); }

// Nicht kopierbar
GLBuffer(const GLBuffer&) = delete;
GLBuffer& operator=(const GLBuffer&) = delete;

// Verschiebbar
GLBuffer(GLBuffer&& other) noexcept : id_(other.id_) {
    other.id_ = 0;
}
GLBuffer& operator=(GLBuffer&& other) noexcept {
    if (this != &other) {
        free();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

// Neuer Konstruktor: explizit aus GLuint
explicit GLBuffer(GLuint id) noexcept : id_(id) {}

void create() {
    free();
    glGenBuffers(1, &id_);
    if (id_ == 0)
        throw std::runtime_error("Hermelin: glGenBuffers failed");
}

void free() {
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
        id_ = 0;
    }
}

GLuint id() const { return id_; }

explicit operator bool() const { return id_ != 0; }

private:
GLuint id_;
};

} // namespace Hermelin
