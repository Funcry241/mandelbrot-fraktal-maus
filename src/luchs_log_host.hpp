// Datei: src/luchs_log_host.hpp
// 🐭 Maus-Kommentar: Nur für Hostcode. Keine CUDA-Arch-Makros. Klar, explizit, fehlerfrei.
// 🦦 Otter: CUDA_CHECK jetzt integraler Bestandteil. Kein Exit, kein fprintf.
// 🦊 Schneefuchs: Host bleibt Host – aber sichtbar. Fehlerpfad klar.

#pragma once
#include <cstdarg>
#include <cuda_runtime.h>
#include <stdexcept> // 🦊 Schneefuchs: Notwendig für std::runtime_error im Hostcode

namespace LuchsLogger {
    void logMessage(const char* file, int line, const char* fmt, ...);
    void flushLogs();
}

// 🦾 Logging-Makro: Variadisch, immer mit Dateipfad + Zeile
#define LUCHS_LOG_HOST(...) LuchsLogger::logMessage(__FILE__, __LINE__, __VA_ARGS__)

// 🧪 CUDA-Fehlermakro – deterministisch, ASCII-only, kein stderr
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            LUCHS_LOG_HOST("[CUDA ERROR] %s failed at %s:%d -> %s",             \
                           #expr, __FILE__, __LINE__, cudaGetErrorString(err__)); \
            throw std::runtime_error("CUDA failure: " #expr);                  \
        }                                                                      \
    } while (0)
#endif
