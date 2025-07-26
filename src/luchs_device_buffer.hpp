#pragma once
#ifndef LUCHS_DEVICE_BUFFER_HPP
#define LUCHS_DEVICE_BUFFER_HPP

#include <cuda_runtime.h>

namespace LuchsLogger {

    // Maximale Größe des globalen Device-Logpuffers (1 MB Standard)
    constexpr int LOG_BUFFER_SIZE = 1 << 20;

    // =========================================================================
    // 🧠 Device-API (wird im __device__-Code verwendet)
    // =========================================================================

    /// Schreibt eine Logzeile in den Device-Puffer (kein printf, keine Formatierung!)
    __device__ void deviceLog(const char* file, int line, const char* msg);

    // =========================================================================
    // 🖥️ Host-API
    // =========================================================================

    /// Setzt Device-Log-Offset und Puffer zurück (zu Beginn eines Frames)
    void resetDeviceLog();

    /// Kopiert den Device-Puffer asynchron in den Host und gibt ihn aus (stderr)
    void flushDeviceLogToHost(cudaStream_t stream);

} // namespace LuchsLogger

#endif // LUCHS_DEVICE_BUFFER_HPP
