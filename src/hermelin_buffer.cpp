///// Otter: RAII für CUDA/GL-Buffer; deterministische ASCII-Logs; Only-Grow-Semantik.
///// Schneefuchs: State-Restore um GL-Binds; /WX-fest; Header/Source synchron – keine Drift.
///// Maus: Implementierung klar getrennt vom Interface; keine versteckten Allokationen.
///// Datei: src/hermelin_buffer.cpp

#include "hermelin_buffer.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#include <GL/glew.h>
#include <cuda_runtime_api.h>   // cudaMalloc / cudaFree
#include <stdexcept>
#include <limits>
#include <utility>              // std::exchange

namespace Hermelin {

// ----------------------------- CUDA-Device-Buffer ----------------------------

CudaDeviceBuffer::CudaDeviceBuffer() : ptr_(nullptr), sizeBytes_(0) {}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    free();
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr))
    , sizeBytes_(std::exchange(other.sizeBytes_, 0)) {}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_       = std::exchange(other.ptr_, nullptr);
        sizeBytes_ = std::exchange(other.sizeBytes_, 0);
    }
    return *this;
}

void CudaDeviceBuffer::allocate(size_t sizeBytes) {
    free();
    if (sizeBytes == 0) {
        sizeBytes_ = 0;
        return;
    }
    const cudaError_t err = cudaMalloc(&ptr_, sizeBytes);
    if (err != cudaSuccess) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CUDA] cudaMalloc failed code=%d bytes=%zu", static_cast<int>(err), sizeBytes);
        }
        ptr_ = nullptr;
        sizeBytes_ = 0;
        throw std::runtime_error("Hermelin: cudaMalloc failed");
    }
    sizeBytes_ = sizeBytes;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CUDA] cudaMalloc ok ptr=%p bytes=%zu", ptr_, sizeBytes_);
    }
}

void CudaDeviceBuffer::resize(size_t sizeBytes) {
    // exakt wie allocate (bewusst kein Copy)
    if (sizeBytes_ == sizeBytes && ptr_ != nullptr) return; // no-op
    allocate(sizeBytes);
}

void CudaDeviceBuffer::ensure(size_t minBytes) {
    // nur wachsen, niemals schrumpfen
    if (sizeBytes_ >= minBytes) return;
    allocate(minBytes);
}

void CudaDeviceBuffer::free() {
    if (ptr_) {
        const void*  p  = ptr_;
        const size_t sz = sizeBytes_;
        const cudaError_t err = cudaFree(ptr_);
        ptr_ = nullptr;
        sizeBytes_ = 0;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CUDA] cudaFree ptr=%p bytes=%zu code=%d", p, sz, static_cast<int>(err));
        }
    }
}

void* CudaDeviceBuffer::get() const {
    return ptr_;
}

size_t CudaDeviceBuffer::size() const {
    return sizeBytes_;
}

// --------------------------------- GL-Buffer ---------------------------------

GLBuffer::GLBuffer() : id_(0) {}

GLBuffer::~GLBuffer() {
    free();
}

GLBuffer::GLBuffer(GLBuffer&& other) noexcept
    : id_(std::exchange(other.id_, 0)) {}

GLBuffer& GLBuffer::operator=(GLBuffer&& other) noexcept {
    if (this != &other) {
        free();
        id_ = std::exchange(other.id_, 0);
    }
    return *this;
}

GLBuffer::GLBuffer(GLuint id) noexcept : id_(id) {}

void GLBuffer::create() {
    free();
    glGenBuffers(1, &id_);
    if (id_ == 0) {
        throw std::runtime_error("Hermelin: glGenBuffers failed");
    }
}

void GLBuffer::allocate(GLsizeiptr sizeBytes, GLenum usage) {
    if (!id_)
        throw std::runtime_error("Hermelin: allocate() called on uninitialized buffer");

    // 🦊 Schneefuchs: State sichern/wiederherstellen, um Seiteneffekte zu vermeiden.
    GLint prev = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev);

    // Sanity: Größenprüfung gegen GLsizeiptr
    if (sizeBytes < 0) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prev));
        throw std::runtime_error("Hermelin: negative size for glBufferData");
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeBytes, nullptr, usage);

    GLint realSize = 0;
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &realSize);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GLBUF] id=%u requested=%lld real=%d usage=0x%X",
                       id_, static_cast<long long>(sizeBytes), realSize, static_cast<unsigned>(usage));
    }

    // Restore previous binding
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prev));

    if (realSize <= 0)
        throw std::runtime_error("Hermelin: glBufferData failed or allocated 0 bytes");
}

void GLBuffer::initAsPixelBuffer(int width, int height, int bytesPerPixel) {
    create();
    // 🦦 Otter: Overflow-sicheres Größenprodukt und sinnvoller Default-Usage für Upload (UNPACK).
    const unsigned long long w   = static_cast<unsigned long long>(width);
    const unsigned long long h   = static_cast<unsigned long long>(height);
    const unsigned long long bpp = static_cast<unsigned long long>(bytesPerPixel);
    const unsigned long long total = w * h * bpp;

    if (w == 0ull || h == 0ull || bpp == 0ull)
        throw std::runtime_error("Hermelin: initAsPixelBuffer invalid dimensions");

    if (total > static_cast<unsigned long long>(std::numeric_limits<GLsizeiptr>::max()))
        throw std::runtime_error("Hermelin: pixel buffer size exceeds GLsizeiptr");

    allocate(static_cast<GLsizeiptr>(total), GL_STREAM_DRAW);
}

void GLBuffer::free() {
    if (id_ != 0) {
        const GLuint del = id_;
        glDeleteBuffers(1, &id_);
        id_ = 0;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[GLBUF] deleted id=%u", del);
        }
    }
}

GLuint GLBuffer::id() const {
    return id_;
}

} // namespace Hermelin
