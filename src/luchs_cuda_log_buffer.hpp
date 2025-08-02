// Datei: src/luchs_cuda_log_buffer.hpp
// 🐭 Maus-Kommentar: Rückkehr zur Einfachheit – Klartext-Logging, jetzt mit `constexpr` statt Makro.
// 🦦 Otter: Kein Makro-Murks mehr – stabil, sicher, sichtbar in allen Translation Units.
// 🦊 Schneefuchs: Determinismus durch statisch geprüfte Größen – kein Bitmüll, keine Zufälligkeit.

#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

// =========================================================================
// 🔒 Konfiguration des CUDA-Logbuffers
// =========================================================================
static constexpr size_t LOG_BUFFER_SIZE = 1024 * 1024; // 1 MB Logpuffer (Empfehlung: 128 KB – 2 MB)

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

    // -------------------------
    // 🐾 Luchs Baby Architektur: Initialisierungskontrolle
    // -------------------------

    // Initialisiert den CUDA-Logpuffer, muss vor flushDeviceLogToHost aufgerufen werden!
    void initCudaLogBuffer(cudaStream_t stream);

    // Gibt Ressourcen frei, falls nötig (optional)
    void freeCudaLogBuffer();

    // Prüft, ob der Logbuffer initialisiert ist
    bool isCudaLogBufferInitialized();

} // namespace LuchsLogger
