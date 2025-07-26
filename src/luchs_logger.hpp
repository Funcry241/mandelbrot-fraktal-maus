// Datei: src/luchs_logger.hpp
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

// CUDA-seitiges Logging (deviceLog)
#ifdef __CUDACC__
__device__ void deviceLog(const char* file, int line, const char* msg);
void resetDeviceLog();
void flushDeviceLogToHost(cudaStream_t stream);
#endif

} // namespace LuchsLogger

// 🔁 Gemeinsames Logging-Makro – entscheidet zur Compilezeit
#ifdef __CUDACC__
#define LUCHS_LOG(msg) ::LuchsLogger::deviceLog(__FILE__, __LINE__, msg)
#else
#define LUCHS_LOG(msg) ::LuchsLogger::logMessage(__FILE__, __LINE__, msg)
#endif

#endif // LUCHS_LOGGER_HPP
