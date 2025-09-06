///// Otter: 64-bit device log offset; simple ASCII; public API unchanged; constexpr sizes only.
///// Schneefuchs: Deterministisch & thread-safe via atomic reservation; no varargs on device; host-only formatting.
///// Maus: CUDA 13-ready; zero UB; single source of truth for buffer size; clean TU separation.
///// Datei: src/luchs_cuda_log_buffer.hpp

#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstddef>
#include <cstdio>

// ============================================================================
// Configuration of the CUDA device log buffer
// ============================================================================
static constexpr size_t LOG_BUFFER_SIZE = 1024 * 1024; // 1 MB device log buffer
static constexpr size_t LOG_MESSAGE_MAX = 512;         // hard cap per single device log line

// ============================================================================
// Namespace: LuchsLogger for device logging
// ============================================================================
namespace LuchsLogger {

    // Device-side call, stores message into the device log buffer (ASCII)
    __device__ void deviceLog(const char* file, int line, const char* msg);

    // Kernel to reset the log buffer (internal use)
    __global__ void resetLogKernel();

    // Host-side: reset the device log buffer
    void resetDeviceLog();

    // Host-side: transfer device buffer via stream to host and print through LUCHS_LOG_HOST
    void flushDeviceLogToHost(cudaStream_t stream);

    // Convenience without stream — uses default stream (0)
    inline void flushDeviceLogToHost() {
        flushDeviceLogToHost(0);
    }

    // -------------------------
    // Luchs Baby architecture: initialization control
    // -------------------------

    // Initialize the CUDA log buffer; must be called before flushDeviceLogToHost
    void initCudaLogBuffer(cudaStream_t stream);

    // Free resources if needed (optional)
    void freeCudaLogBuffer();

    // Check whether the log buffer is initialized
    bool isCudaLogBufferInitialized();

} // namespace LuchsLogger
