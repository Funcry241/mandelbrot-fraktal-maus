// src/utils/cuda_utils.cpp

#include "utils/cuda_utils.hpp"
#include <iostream>
#include <cstdlib>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
