// Datei: src/common.hpp
#pragma once
#include <stdexcept>
#include <cuda_runtime.h>

inline void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}
