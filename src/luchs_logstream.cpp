// Datei: src/luchs_logstream.cpp

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace LuchsLogger {

// Gibt aktuellen Zeitstempel zurück (für logger.cpp)
std::string getTimestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t_c = system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_s(&tm_buf, &t_c);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

} // namespace LuchsLogger
