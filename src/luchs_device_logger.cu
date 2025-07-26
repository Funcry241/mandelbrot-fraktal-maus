#include "luchs_device_buffer.hpp"
#include "luchs_logger.hpp"
#include <cstdio>
#include <cstring>
#include <ctime>

namespace LuchsLogger {

    // =========================================================================
    // 🌌 Device-seitiger Logpuffer (1 MB) + Offset
    // =========================================================================

    __device__ char d_logBuffer[LOG_BUFFER_SIZE];
    __device__ int d_logOffset = 0;

    // Hostseitiger Zwischenspeicher
    static char h_logBuffer[LOG_BUFFER_SIZE] = {0};

    // =========================================================================
    // 🚀 Device-Logfunktion (wird vom Makro LUCHS_LOG im __device__-Code gerufen)
    // =========================================================================

    __device__ void deviceLog(const char* file, int line, const char* msg) {
        int idx = atomicAdd(&d_logOffset, 0);  // Nur lesen
        if (idx >= LOG_BUFFER_SIZE - 128) return;

        int len = 0;

        // Dateiname kopieren
        for (int i = 0; file[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
            d_logBuffer[idx + len++] = file[i];

        // ":<line> | " anhängen
        if (len + 6 + idx < LOG_BUFFER_SIZE) {
            d_logBuffer[idx + len++] = ':';
            int l = line, div = 10000;
            bool started = false;
            for (; div > 0; div /= 10) {
                int digit = (l / div) % 10;
                if (digit != 0 || started || div == 1) {
                    d_logBuffer[idx + len++] = '0' + digit;
                    started = true;
                }
            }
            d_logBuffer[idx + len++] = ' ';
            d_logBuffer[idx + len++] = '|';
            d_logBuffer[idx + len++] = ' ';
        }

        // Nachricht kopieren
        for (int i = 0; msg[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
            d_logBuffer[idx + len++] = msg[i];

        d_logBuffer[idx + len++] = '\n';
        d_logBuffer[idx + len] = 0;

        atomicAdd(&d_logOffset, len);
    }

    // =========================================================================
    // 🧹 Logbuffer zurücksetzen (via Kernel)
    // =========================================================================

    __global__ void resetLogKernel() {
        d_logOffset = 0;
        if (threadIdx.x == 0 && blockIdx.x == 0)
            d_logBuffer[0] = 0;
    }

    void resetDeviceLog() {
        resetLogKernel<<<1,1>>>();
        cudaDeviceSynchronize();
    }

    // =========================================================================
    // 📤 Host: Device-Log auslesen und weiterleiten
    // =========================================================================

    void flushDeviceLogToHost(cudaStream_t stream) {
        cudaMemcpyAsync(h_logBuffer, d_logBuffer, LOG_BUFFER_SIZE, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        h_logBuffer[LOG_BUFFER_SIZE - 1] = '\0';

        const char* line = std::strtok(h_logBuffer, "\n");
        while (line) {
            logMessage("cuda_device_log", 0, line);
            line = std::strtok(nullptr, "\n");
        }
    }

} // namespace LuchsLogger
