///// Datei: src/nacktmull_prog.cuh
#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>

// Progressive-Status (__constant__) + Setter (API unverändert)
struct NacktmullProgState {
    float2*   z;
    uint16_t* it;
    int       addIter;
    int       iterCap;
    int       enabled;
};

extern __device__ __constant__ NacktmullProgState g_prog;

extern "C" void nacktmull_set_progressive(const void* zDev,
                                          const void* itDev,
                                          int addIter, int iterCap, int enabled) noexcept;
