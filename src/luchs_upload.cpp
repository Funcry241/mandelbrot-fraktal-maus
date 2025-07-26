// Datei: src/luchs_upload.cpp
// 🐭 Maus-Kommentar: C4996 beseitigt – strtok_s ersetzt strtok für sichere Zeilenzerlegung. Otter: MSVC-konform, Schneefuchs: threadsafe.

#include "luchs_logger.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

namespace LuchsUpload {

// 🔧 Feste Puffergröße pro Frame (z. B. 8 KB)
constexpr size_t LOG_BUFFER_SIZE = 8192;

static char* d_logBuffer = nullptr;
static char* h_logBuffer = nullptr;

void initCudaLogBuffer() {
    if (!d_logBuffer)
        cudaMalloc(&d_logBuffer, LOG_BUFFER_SIZE);
    if (!h_logBuffer)
        h_logBuffer = new char[LOG_BUFFER_SIZE];
}

void freeCudaLogBuffer() {
    if (d_logBuffer) cudaFree(d_logBuffer);
    d_logBuffer = nullptr;

    delete[] h_logBuffer;
    h_logBuffer = nullptr;
}

char* getDeviceBuffer() {
    return d_logBuffer;
}

void uploadAndPrintLog(cudaStream_t stream) {
    if (!d_logBuffer || !h_logBuffer) return;

    // Asynchrone Kopie aus Device-Logpuffer
    cudaMemcpyAsync(h_logBuffer, d_logBuffer, LOG_BUFFER_SIZE, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Zerlegen in Zeilen (strtok_s für sichere Nutzung)
    h_logBuffer[LOG_BUFFER_SIZE - 1] = '\0'; // Sicherheit
    char* next_token = nullptr;
    char* line = strtok_s(h_logBuffer, "\n", &next_token);
    while (line) {
        LuchsLogger::logMessage("cuda_device_log", 0, line); // Kein konkreter Zeilenbezug
        line = strtok_s(nullptr, "\n", &next_token);
    }
}

} // namespace LuchsUpload
