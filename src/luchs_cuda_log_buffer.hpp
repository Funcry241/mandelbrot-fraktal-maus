// Datei: src/luchs_cuda_log_buffer.hpp
// 🐭 Maus-Kommentar: Otter hat die Device-Loggingstruktur generalisiert. Schneefuchs sorgt für deterministisches Pufferhandling. Alpha 63.

#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

// =========================================================================
// 🔒 Konfiguration des CUDA-Logbuffers
// =========================================================================
#define LOG_BUFFER_SIZE 1048576 // 1 MB Logpuffer (Empfehlung: 128 KB – 2 MB)

// =========================================================================
// 🧠 Device-Logging Makro (verwendbar in __device__ Funktionen)
// =========================================================================
#define LUCHS_LOG_DEVICE(msg) LuchsLogger::deviceLog(__FILE__, __LINE__, msg)

// =========================================================================
// 🧵 Namespace: LuchsLogger für Device-Logging
// =========================================================================
namespace LuchsLogger {

    // Device-seitiger Funktionsaufruf, speichert Nachricht im Puffer
    __device__ void deviceLog(const char* file, int line, const char* msg);

    // Kernel zum Zurücksetzen des Puffers (nur intern verwendet)
    __global__ void resetLogKernel();

    // Host-seitig: Löscht den Logpuffer auf dem Device
    void resetDeviceLog();

    // Host-seitig: Überträgt den Device-Puffer in die stderr-Ausgabe
    void flushDeviceLogToHost(cudaStream_t stream);

} // namespace LuchsLogger
