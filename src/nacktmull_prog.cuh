///// Otter: Deklariert Progressive-State (g_prog) + Setter-API.
///// Schneefuchs: Header-only Deklarationen; keine Seiteneffekte; basisnahe Includes.
///// Maus: C-Schnittstelle; deterministisch; klein.
///// Datei: src/nacktmull_prog.cuh

#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>

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
