// Datei: src/luchs_device_logger.cu
// 🐭 Maus-Kommentar: Diese Datei ist das CUDA-Gegenstück zu LuchsLogger – deviceLog() schreibt in gemeinsamen Buffer. Otter: klare Trennung von Upload- und Device-Logik. Schneefuchs: saubere Ownership-Übergabe.

#include "luchs_logger.hpp"
#include <cstdio>
#include <cstring>
#include <ctime>

namespace LuchsLogger {

// 🔧 Globaler Log-Puffer im Device-Speicher
__device__ char d_logBuffer[LOG_BUFFER_SIZE];
__device__ int d_logOffset = 0;

// 📥 Host-seitiger Empfangs-Puffer
static char h_logBuffer[LOG_BUFFER_SIZE] = {0};

// 🔹 CUDA-Logschreibfunktion (von Device aus aufrufbar)
__device__ void deviceLog(const char* file, int line, const char* msg) {
    int idx = atomicAdd(&d_logOffset, 0); // Vorab prüfen
    if (idx >= LOG_BUFFER_SIZE - 128) return;

    int len = 0;

    // 🔠 Datei + Zeile formatieren
    for (int i = 0; file[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
        d_logBuffer[idx + len++] = file[i];

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

    for (int i = 0; msg[i] && len + idx < LOG_BUFFER_SIZE - 2; ++i)
        d_logBuffer[idx + len++] = msg[i];

    d_logBuffer[idx + len++] = '\n';
    d_logBuffer[idx + len] = 0;

    atomicAdd(&d_logOffset, len);
}

// 🔁 Kernel zur Buffer-Rücksetzung (ein Block reicht)
__global__ void resetLogKernel() {
    d_logOffset = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        d_logBuffer[0] = 0;
}

// 🌐 Hostseitige Reset-API
void resetDeviceLog() {
    resetLogKernel<<<1,1>>>();
    cudaDeviceSynchronize();
}

// 📤 Kopiert den Device-Logbuffer in den Host-Puffer
void downloadDeviceLog(cudaStream_t stream) {
    cudaMemcpyAsync(h_logBuffer, d_logBuffer, LOG_BUFFER_SIZE, cudaMemcpyDeviceToHost, stream);
}

// 📣 Leitet die empfangenen Log-Zeilen an LuchsLogger weiter
void flushDeviceLogToHost(cudaStream_t stream) {
    downloadDeviceLog(stream);
    cudaStreamSynchronize(stream);

    h_logBuffer[LOG_BUFFER_SIZE - 1] = '\0';
    const char* line = std::strtok(h_logBuffer, "\n");
    while (line) {
        logMessage("cuda_device_log", 0, line);
        line = std::strtok(nullptr, "\n");
    }
}

} // namespace LuchsLogger
