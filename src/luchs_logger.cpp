// Datei: src/luchs_logger.cpp
// 🐭 Maus-Kommentar: Host-Implementierung des Logging-Subsystems.
// Otter: Zeitstempel, thread-safe Ausgabe. Schneefuchs: Kein std::cout, nur fprintf(stdout).

#include "luchs_logger.hpp"
#include <chrono>
#include <ctime>
#include <mutex>

namespace LuchsLogger {

    namespace {
        std::mutex logMutex;
    }

    void logMessage(const char* file, int line, const char* msg) {
        std::lock_guard<std::mutex> lock(logMutex);

        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

        std::tm tm_struct{};
        localtime_s(&tm_struct, &t);

        std::fprintf(stdout, "[%02d:%02d:%02d.%03lld] %s:%d: %s\n",
                     tm_struct.tm_hour,
                     tm_struct.tm_min,
                     tm_struct.tm_sec,
                     static_cast<long long>(ms.count()),
                     file,
                     line,
                     msg);
        std::fflush(stdout);
    }

    void flushLogs() {
        std::fflush(stdout);
    }

} // namespace LuchsLogger
