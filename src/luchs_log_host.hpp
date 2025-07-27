// Datei: src/luchs_log_host.hpp
// 🐭 Maus-Kommentar: Nur für Hostcode. Keine CUDA-Arch-Makros. Klar, explizit, fehlerfrei.

#pragma once
#include <cstdarg>

namespace LuchsLogger {
    void logMessage(const char* file, int line, const char* fmt, ...);
    void flushLogs();
}

#define LUCHS_LOG_HOST(...) LuchsLogger::logMessage(__FILE__, __LINE__, __VA_ARGS__)
