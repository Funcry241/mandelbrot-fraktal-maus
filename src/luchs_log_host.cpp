// MAUS:
// Datei: src/luchs_log_host.cpp
// 🐭 Maus-Kommentar: Host-Logging – präzise Zeitstempel, ASCII-only.
// 🦦 Otter: Konsistentes Format Host/Device.
// 🦊 Schneefuchs: Thread-safe, /WX-fest, kein strncat.

#include "luchs_log_host.hpp"
#include "common.hpp" // getLocalTime(...)
#include <chrono>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h> // OutputDebugStringA
#endif

namespace LuchsLogger {
    namespace {
        std::mutex g_logMutex;
    #if defined(_WIN32)
        bool g_mirrorToDebugger = true;  // default: mirror logs to debugger
    #else
        bool g_mirrorToDebugger = false;
    #endif
    } // anon

    void setMirrorToDebugger(bool enable) noexcept { g_mirrorToDebugger = enable; }

    void logMessage(const char* file, int line, const char* fmt, ...) {
        std::lock_guard<std::mutex> guard(g_logMutex);

        // Filename only (strip path)
        const char* base = std::strrchr(file, '\\');
        if (!base) base = std::strrchr(file, '/');
        base = base ? base + 1 : file;

        // Timestamp (YYYY-MM-DD HH:MM:SS.mmm)
        const auto now = std::chrono::system_clock::now();
        const auto t   = std::chrono::system_clock::to_time_t(now);
        const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now.time_since_epoch()) % 1000;

        std::tm tm{};
        getLocalTime(tm, t);

        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);

        // ----- stdout pass -----
        std::fprintf(stdout, "[%s.%03lld][%s][%d]: ",
                     ts, static_cast<long long>(ms.count()), base, line);

        va_list a1;
        va_start(a1, fmt);
        std::vfprintf(stdout, fmt, a1);
        va_end(a1);

        std::fputc('\n', stdout);
        std::fflush(stdout);

        // ----- optional debugger mirror (Windows) -----
    #if defined(_WIN32)
        if (g_mirrorToDebugger) {
            char buf[2048];
            int n = std::snprintf(buf, sizeof(buf), "[%s.%03lld][%s][%d]: ",
                                  ts, static_cast<long long>(ms.count()), base, line);
            if (n < 0) n = 0;
            if (n >= static_cast<int>(sizeof(buf))) n = static_cast<int>(sizeof(buf)) - 1;

            va_list a2;
            va_start(a2, fmt);
            int m = 0;
            if (n < static_cast<int>(sizeof(buf))) {
                m = std::vsnprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), fmt, a2);
                if (m < 0) m = 0;
            }
            va_end(a2);

            // Safe newline append (kein strncat / keine „unsafe“ CRTs)
            size_t len = std::strlen(buf);
            if (len + 1 < sizeof(buf)) {
                buf[len] = '\n';
                buf[len + 1] = '\0';
            } else {
                buf[sizeof(buf) - 2] = '\n';
                buf[sizeof(buf) - 1] = '\0';
            }

            OutputDebugStringA(buf);
        }
    #endif
    }

    void flushLogs() { std::fflush(stdout); }

} // namespace LuchsLogger
