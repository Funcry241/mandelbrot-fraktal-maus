// Datei: src/memory_utils.cu
// 🐭 Maus-Kommentar: Exception-basiertes Fehlerhandling bei Speicher-Allokation

#include <cuda_runtime.h>
#include <stdexcept>   // 🐭 Besser: <stdexcept> für Exception
#include "memory_utils.hpp"

// 🐭 Device-Speicher für Complexity-Buffer mit sauberem Fehler-Handling
extern "C" float* allocComplexityBuffer(int totalTiles) {
    float* d_complexity = nullptr;
    cudaError_t err = cudaMalloc(&d_complexity, totalTiles * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc für Complexity-Buffer fehlgeschlagen: ") +
            cudaGetErrorString(err)
        );
    }
    return d_complexity;
}
