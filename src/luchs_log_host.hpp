// MAUS:
// Datei: src/luchs_log_host.hpp
// 🐭 Maus: Host-only Logging, klarer Vertrag, ASCII-only.
// 🦦 Otter: CUDA_CHECK integriert, deterministischer Fehlerpfad. (Bezug zu Otter)
// 🦊 Schneefuchs: Keine Header-Seiteneffekte, /WX-fest. (Bezug zu Schneefuchs)
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
#define LUCHS_LOG_HOST(...) ::LuchsLogger::logMessage(__FILE__, __LINE__, __VA_ARGS__)

// CUDA error check — ASCII-only & throws, no stderr side-effects.
// Guarded to avoid redefinition if included multiple times.
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                           \
    do {                                                                           \
        cudaError_t err__ = (expr);                                                \
        if (err__ != cudaSuccess) {                                                \
            const char* _msg = ::cudaGetErrorString(err__);                        \
            if (!_msg) _msg = "<cudaGetErrorString=null>";                         \
            LUCHS_LOG_HOST("[CUDA ERROR] %s failed -> %s", #expr, _msg);           \
            throw std::runtime_error("CUDA failure: " #expr);                      \
        }                                                                          \
    } while (0)
#endif
