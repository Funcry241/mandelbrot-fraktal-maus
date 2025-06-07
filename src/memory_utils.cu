// Datei: src/memory_utils.cu
// 🐭 Maus-Kommentar: CUDA-Speicher-Handling mit brutal klarer Exception

#include <cuda_runtime.h>
#include <stdexcept>
#include "memory_utils.hpp"

extern "C" float* allocComplexityBuffer(int totalTiles) {
    float* d = nullptr;
    if (auto err = cudaMalloc(&d, totalTiles * sizeof(float)); err != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    return d;
}
