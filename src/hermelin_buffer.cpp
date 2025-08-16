// Datei: src/hermelin_buffer.cpp
// 🐜 Hermelin: Implementation der RAII-Buffer für CUDA- und OpenGL-Ressourcen.
// 🦦 Otter: Trennung von Interface und Logik. Fehler-Logging explizit, Speichergrößen sichtbar.
// 🦊 Schneefuchs: Keine Logik im Header. Alles klar nachvollziehbar.

#include "hermelin_buffer.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"
#include <GL/glew.h>
#include <stdexcept>

namespace Hermelin {

// --- CUDA-Device-Buffer ---

CudaDeviceBuffer::CudaDeviceBuffer() : ptr_(nullptr), sizeBytes_(0) {}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    free();
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), sizeBytes_(other.sizeBytes_) {
    other.ptr_ = nullptr;
    other.sizeBytes_ = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        sizeBytes_ = other.sizeBytes_;
        other.ptr_ = nullptr;
        other.sizeBytes_ = 0;
    }
    return *this;
}

void CudaDeviceBuffer::allocate(size_t sizeBytes) {
    free();
    cudaError_t err = cudaMalloc(&ptr_, sizeBytes);
    if (err != cudaSuccess) {
        ptr_ = nullptr;
        sizeBytes_ = 0;
        throw std::runtime_error("Hermelin: cudaMalloc failed");
    }
    sizeBytes_ = sizeBytes;
}

void CudaDeviceBuffer::free() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        sizeBytes_ = 0;
    }
}

void* CudaDeviceBuffer::get() const {
    return ptr_;
}

size_t CudaDeviceBuffer::size() const {
    return sizeBytes_;
}

// --- OpenGL-Buffer ---

GLBuffer::GLBuffer() : id_(0) {}

GLBuffer::~GLBuffer() {
    free();
}

GLBuffer::GLBuffer(GLBuffer&& other) noexcept
    : id_(other.id_) {
    other.id_ = 0;
}

GLBuffer& GLBuffer::operator=(GLBuffer&& other) noexcept {
    if (this != &other) {
        free();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

GLBuffer::GLBuffer(GLuint id) noexcept : id_(id) {}

void GLBuffer::create() {
    free();
    glGenBuffers(1, &id_);
    if (id_ == 0)
        throw std::runtime_error("Hermelin: glGenBuffers failed");
}

void GLBuffer::allocate(GLsizeiptr sizeBytes, GLenum usage) {
    if (!id_)
        throw std::runtime_error("Hermelin: allocate() called on uninitialized buffer");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeBytes, nullptr, usage);

    GLint realSize = 0;
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &realSize);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GLBUF] ID %u: Requested %zd bytes, GL reports %d bytes",
                       id_, static_cast<size_t>(sizeBytes), realSize);
    }

    if (realSize <= 0)
        throw std::runtime_error("Hermelin: glBufferData failed or allocated 0 bytes");
}

void GLBuffer::initAsPixelBuffer(int width, int height, int bytesPerPixel) {
    create();
    const size_t size = static_cast<size_t>(width) * height * bytesPerPixel;
    allocate(size);
}

void GLBuffer::free() {
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
        id_ = 0;
    }
}

GLuint GLBuffer::id() const {
    return id_;
}

} // namespace Hermelin
