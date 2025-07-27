// Datei: src/luchs_cuda_log_buffer.hpp
// 🐭 Maus-Kommentar: Rückkehr zur Einfachheit – Klartext-Logging statt variadisch. Kein vsnprintf im __device__-Code.
// 🦦 Otter: Formatierung raus, Sicherheit rein. Header konsistent zur .cu-Implementierung.
// 🦊 Schneefuchs: Determinismus durch Reduktion – Formatierungsfreiheit auf CUDA-Level.

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
//     Nur Klartext – keine Formatierung im __device__-Code
// =========================================================================
#define LUCHS_LOG_DEVICE(msg) LuchsLogger::deviceLog(__FILE__, __LINE__, msg)

// =========================================================================
// 🧵 Namespace: LuchsLogger für Device-Logging
// =========================================================================
namespace LuchsLogger {

    // Device-seitiger Funktionsaufruf, speichert Nachricht im Logpuffer
    __device__ void deviceLog(const char* file, int line, const char* msg);

    // Kernel zum Zurücksetzen des Logpuffers (nur intern verwendet)
    __global__ void resetLogKernel();

    // Host-seitig: Löscht den Logpuffer auf dem Device
    void resetDeviceLog();

    // Host-seitig: Überträgt den Device-Puffer via Stream auf den Host
    void flushDeviceLogToHost(cudaStream_t stream);

    // 🦦 Otter: Convenience-Funktion ohne Stream – nutzt Default-Stream (0)
    inline void flushDeviceLogToHost() {
        flushDeviceLogToHost(0);
    }

} // namespace LuchsLogger
