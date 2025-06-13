// Datei: src/memory_utils.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>   // ✨ Fix: für std::exit()
#include "memory_utils.hpp"

namespace MemoryUtils {

// Device-Speicher für Complexity-Buffer allokieren
float* allocComplexityBuffer(int totalTiles) {
    float* d_complexity = nullptr;
    cudaError_t err = cudaMalloc(&d_complexity, totalTiles * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc für Complexity-Buffer fehlgeschlagen: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    return d_complexity;
}

// Device-Speicher freigeben (idempotent)
void freeComplexityBuffer(float*& d_buffer) {
    if (d_buffer) {
        cudaFree(d_buffer);  // Fehler werden bewusst ignoriert
        d_buffer = nullptr;
    }
}

} // namespace MemoryUtils
