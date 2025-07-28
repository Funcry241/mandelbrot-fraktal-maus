// Datei: src/luchs_logger.cpp
// 🐭 Maus-Kommentar: Host-Logging mit präzisem Zeitformat und konsistentem Stil für alle Plattformen.
// Otter: Gleiches Format wie Device. Schneefuchs: Keine Zeitabweichungen mehr!

#include "luchs_log_host.hpp"
#include "common.hpp" // 🦊 Schneefuchs: getLocalTime(...) für plattformübergreifende Zeit
#include <chrono>
#include <mutex>
#include <cstdarg>

namespace LuchsLogger {

    namespace {
        std::mutex logMutex;
    }

    void logMessage(const char* file, int line, const char* fmt, ...) {
        std::lock_guard<std::mutex> lock(logMutex);

        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;

        std::tm tm_struct{};
        getLocalTime(tm_struct, t); // 🦊 Schneefuchs: Plattformübergreifend, threadsicher

        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm_struct);

        std::fprintf(stdout, "[%s.%03lld][%s][%d]: ",
                     timebuf,
                     static_cast<long long>(ms.count()),
                     file,
                     line);

        va_list args;
        va_start(args, fmt);
        std::vfprintf(stdout, fmt, args);
        va_end(args);

        std::fprintf(stdout, "\n");
        std::fflush(stdout);
    }

    void flushLogs() {
        std::fflush(stdout);
    }

} // namespace LuchsLogger
