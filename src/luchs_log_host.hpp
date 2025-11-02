///// Otter: Host-only logging – single source for CUDA_CHECK (+ non-throwing variant); deterministic, ASCII-only
///// Schneefuchs: Header has no side effects; /WX-safe; one macro home; captures file/line for every log
///// Maus: Uniform format; optional Windows debugger mirror via impl; no hidden deps; stable API
///// Datei: src/luchs_log_host.hpp

#pragma once

#include <cstdarg>
#include <cuda_runtime.h>    // cudaError_t, cudaGetErrorString
#include <stdexcept>

namespace LuchsLogger {
    // Thread-safe host logger with uniform formatting.
    void logMessage(const char* file, int line, const char* fmt, ...);
    void flushLogs();

    // Optional: also mirror logs to the Windows debugger (OutputDebugStringA).
    // Default: enabled on Windows, ignored elsewhere.
    void setMirrorToDebugger(bool enable) noexcept;
}

// Variadic convenience macro: captures call site file/line.
#ifndef LUCHS_LOG_HOST
#define LUCHS_LOG_HOST(...) ::LuchsLogger::logMessage(__FILE__, __LINE__, __VA_ARGS__)
#endif

// -----------------------------------------------------------------------------
// CUDA error checks – single canonical home
// -----------------------------------------------------------------------------

// Throwing check: logs ASCII detail and throws std::runtime_error on failure.
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                           \
    do {                                                                           \
        cudaError_t err__ = (expr);                                                \
        if (err__ != cudaSuccess) {                                                \
            const char* _msg = ::cudaGetErrorString(err__);                        \
            if (!_msg) _msg = "<cudaGetErrorString=null>";                         \
            LUCHS_LOG_HOST("[CUDA][ERR] %s -> rc=%d msg=%s", #expr, (int)err__, _msg); \
            throw std::runtime_error("CUDA failure: " #expr);                      \
        }                                                                          \
    } while (0)
#endif

// Non-throwing check: logs ASCII detail and continues (for hot paths / timing).
#ifndef CUDA_CHECK_NT
#define CUDA_CHECK_NT(expr)                                                        \
    do {                                                                           \
        cudaError_t err__ = (expr);                                                \
        if (err__ != cudaSuccess) {                                                \
            const char* _msg = ::cudaGetErrorString(err__);                        \
            if (!_msg) _msg = "<cudaGetErrorString=null>";                         \
            LUCHS_LOG_HOST("[CUDA][WARN] %s -> rc=%d msg=%s", #expr, (int)err__, _msg); \
        }                                                                          \
    } while (0)
#endif
