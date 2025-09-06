///// Otter: 64-bit atomic device logging with reservation; local format then copy; no overlap.
///// Schneefuchs: Uses cuda::atomic_ref (CUDA 13) for thread_scope_device; ASCII-only; deterministic bounds.
///// Maus: Flush copies only written bytes; reset kernel zeroes offset; host/device logs strictly separated.
///// Datei: src/luchs_cuda_log_buffer.cu

#include "luchs_cuda_log_buffer.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#include <cstring>
#include <cuda/atomic>

namespace LuchsLogger {

    // =========================================================================
    // Device-side log storage (1 MB) + 64-bit write offset
    // =========================================================================

    __device__ __align__(8) char d_logBuffer[LOG_BUFFER_SIZE];
    __device__ __align__(8) unsigned long long d_logOffset = 0ULL;

    // Host-side staging buffer
    static char h_logBuffer[LOG_BUFFER_SIZE] = {0};

    // =========================================================================
    // Initialization state and preferred stream (host-side)
    // =========================================================================

    static bool s_isInitialized = false;
    static cudaStream_t s_logStream = nullptr;

    // =========================================================================
    // Device logging — reserve, then copy (no varargs on device)
    // =========================================================================

    __device__ void deviceLog(const char* file, int line, const char* msg) {
        // Compose into a local buffer first to know the exact length.
        char local[LOG_MESSAGE_MAX];
        int len = 0;

        // 1) filename (basename only)
        const char* filenameOnly = file;
        for (int i = 0; file[i] != '\0'; ++i) {
            if (file[i] == '/' || file[i] == '\\') filenameOnly = &file[i + 1];
        }
        for (int i = 0; filenameOnly[i] && len < int(LOG_MESSAGE_MAX) - 1; ++i) {
            local[len++] = filenameOnly[i];
        }

        // 2) ':' line ' ' separator
        if (len < int(LOG_MESSAGE_MAX) - 1) {
            local[len++] = ':';
            // write decimal line number
            int l = (line < 0) ? 0 : line;
            // max 10 digits for 32-bit int
            int digits[10]; int dcount = 0;
            if (l == 0) { digits[dcount++] = 0; }
            while (l > 0 && dcount < 10) { digits[dcount++] = l % 10; l /= 10; }
            for (int k = dcount - 1; k >= 0 && len < int(LOG_MESSAGE_MAX) - 1; --k) {
                local[len++] = char('0' + digits[k]);
            }
            if (len < int(LOG_MESSAGE_MAX) - 1) local[len++] = ' ';
        }

        // 3) delimiter " | "
        if (len + 3 < int(LOG_MESSAGE_MAX)) {
            local[len++] = '|';
            local[len++] = ' ';
        }

        // 4) message
        for (int i = 0; msg[i] && len < int(LOG_MESSAGE_MAX) - 2; ++i) {
            local[len++] = msg[i];
        }

        // 5) newline
        if (len < int(LOG_MESSAGE_MAX) - 1) {
            local[len++] = '\n';
        }

        // Reserve space atomically
        auto ref = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(d_logOffset);
        unsigned long long idx = ref.fetch_add((unsigned long long)len, cuda::memory_order_relaxed);

        // Bounds check and copy
        if (idx >= (unsigned long long)LOG_BUFFER_SIZE) {
            // reservation beyond buffer — drop silently
            return;
        }
        unsigned long long maxCopy = (unsigned long long)LOG_BUFFER_SIZE - idx;
        unsigned long long toCopy  = (unsigned long long)((len <= (int)maxCopy) ? len : (int)maxCopy);

        // Copy byte-wise (simple & portable)
        for (unsigned long long i = 0; i < toCopy; ++i) {
            d_logBuffer[idx + i] = local[i];
        }

        // Attempt to 0-terminate next char if available (benign if overwritten later)
        if (idx + toCopy < (unsigned long long)LOG_BUFFER_SIZE) {
            d_logBuffer[idx + toCopy] = 0;
        }
    }

    // =========================================================================
    // Reset log buffer via kernel
    // =========================================================================

    __global__ void resetLogKernel() {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            d_logOffset = 0ULL;
            d_logBuffer[0] = 0;
        }
    }

    void resetDeviceLog() {
        if (!s_isInitialized) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby ERROR] resetDeviceLog called before init!");
            }
            return;
        }
        resetLogKernel<<<1,1,0,s_logStream>>>();
        CUDA_CHECK(cudaStreamSynchronize(s_logStream));
    }

    // =========================================================================
    // Initialization / teardown
    // =========================================================================

    void initCudaLogBuffer(cudaStream_t stream) {
        if (s_isInitialized) return;
        s_logStream = (stream == nullptr) ? 0 : stream;

        // Clear device state
        resetLogKernel<<<1,1,0,s_logStream>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s_logStream));

        // Clear host staging
        std::memset(h_logBuffer, 0, sizeof(h_logBuffer));

        s_isInitialized = true;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[LuchsBaby] LogBuffer initialized (size=%zu, stream=%p)", (size_t)LOG_BUFFER_SIZE, (void*)s_logStream);
        }
    }

    void freeCudaLogBuffer() {
        if (!s_isInitialized) return;
        // nothing to free (symbols are static), reset for good measure
        resetLogKernel<<<1,1,0,s_logStream>>>();
        (void)cudaGetLastError();
        (void)cudaStreamSynchronize(s_logStream);
        s_isInitialized = false;
        s_logStream = nullptr;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[LuchsBaby] LogBuffer freed");
        }
    }

    bool isCudaLogBufferInitialized() {
        return s_isInitialized;
    }

    // =========================================================================
    // Host: fetch device log and print via LUCHS_LOG_HOST
    // =========================================================================

    void flushDeviceLogToHost(cudaStream_t stream) {
        if (!s_isInitialized) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby ERROR] flushDeviceLogToHost called before init!");
            }
            return;
        }

        // Normalize stream
        cudaStream_t useStream = (stream == nullptr) ? s_logStream : stream;

        // 1) Fetch current offset
        unsigned long long used = 0ULL;
        CUDA_CHECK(cudaMemcpyFromSymbolAsync(
            &used, d_logOffset, sizeof(used), 0, cudaMemcpyDeviceToHost, useStream));
        CUDA_CHECK(cudaStreamSynchronize(useStream));

        if (used == 0ULL) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby] flush: no data");
            }
            return;
        }

        // 2) Copy only the used bytes (cap at buffer size - 1)
        size_t toCopy = (used > (unsigned long long)(LOG_BUFFER_SIZE - 1))
                        ? (LOG_BUFFER_SIZE - 1)
                        : (size_t)used;

        CUDA_CHECK(cudaMemcpyFromSymbolAsync(
            h_logBuffer, d_logBuffer, toCopy, 0, cudaMemcpyDeviceToHost, useStream));
        CUDA_CHECK(cudaStreamSynchronize(useStream));

        // ensure 0-termination for safe host-side parsing
        h_logBuffer[toCopy] = 0;

        // 3) Print line-by-line (only when debugLogging to avoid spam)
        if constexpr (Settings::debugLogging) {
            char* ptr = h_logBuffer;
            while (*ptr) {
                char* lineEnd = std::strchr(ptr, '\n');
                if (!lineEnd) break;
                *lineEnd = 0;
                LUCHS_LOG_HOST("[CUDA] %s", ptr);
                ptr = lineEnd + 1;
            }
        }
    }

} // namespace LuchsLogger
