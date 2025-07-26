#include "luchs_buffer.hpp"
#include "luchs_logger.hpp"
#include <cstdio>
#include <cstring>
#include <ctime>

namespace Luchs {

// Logbuffer im globalen Device-Speicher
__device__ char d_logBuffer[LOG_BUFFER_SIZE];
__device__ int d_logOffset = 0;

// Hostseitige Kopie
char h_logBuffer[LOG_BUFFER_SIZE] = {0};

__device__ void deviceLog(const char* file, int line, const char* msg) {
    int idx = atomicAdd(&d_logOffset, 0); // Vorab prüfen: genug Platz?
    if (idx >= LOG_BUFFER_SIZE - 128) return;

    // Format: "<file>:<line> | <msg>\n"
    int len = 0;

    // Schlichtes Formatieren, kein sprintf im Device-Code
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

__global__ void resetLogKernel() {
    d_logOffset = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        d_logBuffer[0] = 0;
}

void resetDeviceLog() {
    resetLogKernel<<<1,1>>>();
    cudaDeviceSynchronize();
}

void downloadLog(cudaStream_t stream) {
    cudaMemcpyAsync(h_logBuffer, d_logBuffer, LOG_BUFFER_SIZE, cudaMemcpyDeviceToHost, stream);
}

void flushLogToConsole() {
    char* ptr = h_logBuffer;
    while (*ptr) {
        char* lineEnd = strchr(ptr, '\n');
        if (!lineEnd) break;
        *lineEnd = 0;

        std::time_t now = time(nullptr);
        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        std::printf("[%-19s] %s\n", timebuf, ptr);        
        ptr = lineEnd + 1;
    }
}

} // namespace Luchs
