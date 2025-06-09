// Datei: src/memory_utils.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>   // ✨ Fix: für std::exit()
#include "memory_utils.hpp"

namespace MemoryUtils { // <--- 🐾 Namespace öffnen!

// Device-Speicher für Complexity-Buffer
float* allocComplexityBuffer(int totalTiles) {
    float* d_complexity = nullptr;
    cudaError_t err = cudaMalloc(&d_complexity, totalTiles * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc für Complexity-Buffer fehlgeschlagen: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    return d_complexity;
}

} // namespace MemoryUtils  // <--- 🐾 Namespace schließen!
