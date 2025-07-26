#pragma once
#include <iostream>
#include <cstdio>
#include <cstdarg>

// Stream-basiertes Logging (LUCHS_LOG_STREAM << ...)
struct LogStream {
    template <typename T>
    LogStream& operator<<(const T& value) {
        std::cerr << value;
        return *this;
    }

    // Für std::endl etc.
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        std::cerr << manip;
        return *this;
    }
};

inline LogStream LUCHS_LOG_STREAM;

// printf-artiges Logging (LUCHS_LOG("x = %d\n", x))
inline void LUCHS_LOG_PRINTF(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

// Makro-Magie: LUCHS_LOG(...) ruft LUCHS_LOG_PRINTF auf
#define LUCHS_LOG(...) LUCHS_LOG_PRINTF(__VA_ARGS__)

// 🧪 Erweiterbar:
// #define LUCHS_LOG_ERR(...)  LUCHS_LOG("[ERROR] " __VA_ARGS__)
// #define LUCHS_LOG_DBG(...)  if (Settings::debugLogging) LUCHS_LOG("[DEBUG] " __VA_ARGS__)
