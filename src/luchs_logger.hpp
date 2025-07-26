// 🐭 Maus-Kommentar: Einheitliches Logging für Host (C++) und Device (CUDA). Host-Ausgabe mit Zeitstempel, Device via Buffer. Otter: UniversalMakro. Schneefuchs: Compilezeit-Switch.

#pragma once
#ifndef LUCHS_LOGGER_HPP
#define LUCHS_LOGGER_HPP

#include <cstdio>

namespace LuchsLogger {

// Host-seitiges Logging mit Zeitstempel und Ursprungsort
void logMessage(const char* file, int line, const char* msg);

// Optional: explizites Flush (falls gepuffert wird)
void flushLogs();

} // namespace LuchsLogger

#ifdef __CUDACC__
// Device-Code → ruft CUDA-Puffer-Logik auf
namespace Luchs {
__device__ void deviceLog(const char* file, int line, const char* msg);
}
#define LUCHS_LOG(msg) Luchs::deviceLog(__FILE__, __LINE__, msg)
#else
// Host-Code → nutzt Zeitstempel-Logging
#define LUCHS_LOG(msg) ::LuchsLogger::logMessage(__FILE__, __LINE__, msg)
#endif

#endif // LUCHS_LOGGER_HPP
