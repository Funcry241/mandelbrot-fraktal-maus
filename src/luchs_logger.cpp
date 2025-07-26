// Datei: src/luchs_logger.cpp

#include "luchs_logger.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace LuchsLogger {

static std::mutex logMutex;

void logMessage(const char* file, int line, const char* msg) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto time_t_now = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count()
        << " [" << file << ":" << line << "] " << msg << '\n';

    std::lock_guard<std::mutex> lock(logMutex);
    std::fputs(oss.str().c_str(), stdout);
    std::fflush(stdout);
}

void flushLogs() {
    std::fflush(stdout);
}

} // namespace LuchsLogger
