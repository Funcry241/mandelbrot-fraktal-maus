///// Otter: Rueckbau auf Klartext-Device-Logging; keine Varargs; robust, simpel, sicher.
///// Schneefuchs: cudaMemcpyFromSymbolAsync statt ungueltiger Symbol-Zeiger; deterministisch & korrekt.
///// Maus: ASCII-only; Host-orchestriert; Logs ausschliesslich via LUCHS_LOG_HOST.
///// Datei: src/luchs_cuda_log_buffer.cu

#include "luchs_cuda_log_buffer.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"
#include <cstring>

namespace LuchsLogger {

    // =========================================================================
    // Device-seitiger Logpuffer (1 MB) + Offset (nur hier definiert)
    // =========================================================================

    __device__ char d_logBuffer[LOG_BUFFER_SIZE];
    __device__ int d_logOffset = 0;

    // Hostseitiger Zwischenspeicher
    static char h_logBuffer[LOG_BUFFER_SIZE] = {0};

    // =========================================================================
    // Initialisierungsstatus und Stream speichern
    // =========================================================================

    static bool s_isInitialized = false;
    static cudaStream_t s_logStream = nullptr;

    // =========================================================================
    // Device-Logfunktion – kein Format, nur Klartext (LUCHS_LOG_DEVICE)
    // =========================================================================

    __device__ void deviceLog(const char* file, int line, const char* msg) {
        int idx = atomicAdd(&d_logOffset, 0);  // nur lesen (aktueller Offset)
        if (idx >= LOG_BUFFER_SIZE - 256) return;

        int len = 0;

        // Dateiname extrahieren (ohne Pfad)
        const char* filenameOnly = file;
        for (int i = 0; file[i] != '\0'; ++i) {
            if (file[i] == '/' || file[i] == '\\')
                filenameOnly = &file[i + 1];
        }

        // "file:line | " schreiben
        for (int i = 0; filenameOnly[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
            d_logBuffer[idx + len++] = filenameOnly[i];

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

        // Nachricht
        for (int i = 0; msg[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
            d_logBuffer[idx + len++] = msg[i];

        // Zeilenende + Nullterminator
        d_logBuffer[idx + len++] = '\n';
        d_logBuffer[idx + len]   = 0;

        // neuen Offset publizieren
        atomicAdd(&d_logOffset, len);
    }

    // =========================================================================
    // Logbuffer zuruecksetzen (via Kernel)
    // =========================================================================

    __global__ void resetLogKernel() {
        d_logOffset = 0;
        if (threadIdx.x == 0 && blockIdx.x == 0)
            d_logBuffer[0] = 0;
    }

    void resetDeviceLog() {
        if (!s_isInitialized) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby ERROR] resetDeviceLog called before init!");
            }
            return;
        }
        resetLogKernel<<<1,1>>>();
        CUDA_CHECK(cudaStreamSynchronize(s_logStream));
    }

    // =========================================================================
    // Initialisierung / Freigabe
    // =========================================================================

    void initCudaLogBuffer(cudaStream_t stream) {
        if (s_isInitialized) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby INFO] initCudaLogBuffer already called.");
            }
            return;
        }
        s_logStream = stream;
        resetLogKernel<<<1,1>>>();
        CUDA_CHECK(cudaStreamSynchronize(s_logStream));
        s_isInitialized = true;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[LuchsBaby] LogBuffer initialized on stream %p", (void*)stream);
        }
    }

    void freeCudaLogBuffer() {
        if (!s_isInitialized) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby INFO] freeCudaLogBuffer called but not initialized.");
            }
            return;
        }
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
    // Host: Device-Logbuffer auslesen und ueber LUCHS_LOG_HOST ausgeben
    // =========================================================================

    void flushDeviceLogToHost(cudaStream_t stream) {
        if (!s_isInitialized) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby ERROR] flushDeviceLogToHost called before init!");
            }
            return;
        }

        // Stream normalisieren
        if (stream == nullptr) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[LuchsBaby] stream==nullptr, using default stream 0");
            }
            stream = 0;
        }

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[DEBUG] flushDeviceLogToHost: using cudaMemcpyFromSymbolAsync (size=%zu)", LOG_BUFFER_SIZE);
        }

        // Pufferinhalt kopieren (Symbol -> Host)
        cudaError_t copyErr = cudaMemcpyFromSymbolAsync(
            h_logBuffer,
            d_logBuffer,
            LOG_BUFFER_SIZE,
            0,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (copyErr != cudaSuccess) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[CUDA ERROR] cudaMemcpyFromSymbolAsync failed: %s", cudaGetErrorString(copyErr));
            }
            return;
        }

        // Warten, bis der Copy fertig ist
        cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[CUDA ERROR] cudaStreamSynchronize failed: %s", cudaGetErrorString(syncErr));
            }
            return;
        }

        // Zeilenweise ausgeben (nur im Debug-Modus, um Log-Spam zu vermeiden)
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
