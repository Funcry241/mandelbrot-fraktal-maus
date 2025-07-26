// Datei: src/luchs_device_buffer.hpp

#pragma once
#ifndef LUCHS_DEVICE_BUFFER_HPP
#define LUCHS_DEVICE_BUFFER_HPP

#include <cuda_runtime.h>

namespace Luchs {

// Maximale Größe des globalen Device-Logpuffers (1 MB Standard)
constexpr int LOG_BUFFER_SIZE = 1 << 20;

// ======= Device-API =======

// Wird im __device__-Code verwendet, um eine Logzeile zu erzeugen
__device__ void deviceLog(const char* file, int line, const char* msg);

// ======= Host-API =======

// Setzt den Logpuffer im Device zurück (Offset und Inhalt)
void resetDeviceLog();

// Kopiert den Device-Logpuffer asynchron in den Hostpuffer
void downloadLog(cudaStream_t stream);

// Gibt den hostseitig übertragenen Logpuffer in der Konsole aus (mit Timestamp)
void flushLogToConsole();

} // namespace Luchs

// ======= Makro für CUDA-Device-Code =======
#ifdef __CUDA_ARCH__
#define LUCHS_DEVICE_LOG(msg) ::Luchs::deviceLog(__FILE__, __LINE__, msg)
#else
#define LUCHS_DEVICE_LOG(msg) ((void)0)
#endif

#endif // LUCHS_DEVICE_BUFFER_HPP
