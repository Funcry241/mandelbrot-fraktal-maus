// Datei: src/luchs_cuda_log_buffer.cu
// 🐭 Maus-Kommentar: Rückbau auf klare Nicht-Formatierung – robust, simpel, sicher.
// 🦦 Otter: Keine varargs mehr – Klartext-only im device-Code, kompatibel & portabel.
// 🦊 Schneefuchs: Präzise Begrenzung, keine Host-Abhängigkeit, garantiert lauffähig.
#include "luchs_cuda_log_buffer.hpp"
#include "luchs_log_host.hpp"
#include <cstring>

namespace LuchsLogger {

    // =========================================================================
    // 🌌 Device-seitiger Logpuffer (1 MB) + Offset (nur hier definiert!)
    // =========================================================================

    __device__ char d_logBuffer[LOG_BUFFER_SIZE];
    __device__ int d_logOffset = 0;

    // Hostseitiger Zwischenspeicher
    char h_logBuffer[LOG_BUFFER_SIZE] = {0};

    // =========================================================================
    // 🚀 Device-Logfunktion – kein Format, nur Klartext (LUCHS_LOG_DEVICE)
    // =========================================================================

    __device__ void deviceLog(const char* file, int line, const char* msg) {
        int idx = atomicAdd(&d_logOffset, 0);  // Nur lesen
        if (idx >= LOG_BUFFER_SIZE - 256) return;

        int len = 0;

        // Rückwärts über das C-String-Ende iterieren
        const char* filenameOnly = file;
        for (int i = 0; file[i] != '\0'; ++i) {
            if (file[i] == '/' || file[i] == '\\')
                filenameOnly = &file[i + 1];
        }

        // Filename schreiben
        for (int i = 0; filenameOnly[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
            d_logBuffer[idx + len++] = filenameOnly[i];

        // ":" + Zeile + " | "
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

        // Nachricht (Klartext)
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
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // =========================================================================
    // 📤 Host: Device-Logbuffer auslesen und über LUCHS_LOG_HOST ausgeben
    // =========================================================================

    void flushDeviceLogToHost(cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(h_logBuffer, d_logBuffer, LOG_BUFFER_SIZE, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        char* ptr = h_logBuffer;
        while (*ptr) {
            char* lineEnd = strchr(ptr, '\n');
            if (!lineEnd) break;
            *lineEnd = 0;

            LUCHS_LOG_HOST("[CUDA] %s", ptr);
            ptr = lineEnd + 1;
        }
    }

} // namespace LuchsLogger
