///// Datei: src/nacktmull_pert.cuh
#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include "core_kernel.h" // PerturbParams, PertStore

// PERT-Params (__constant__) + GLOBAL-Orbit-Pointer + Host-Setter
extern __device__ __constant__ PerturbParams g_pert;
extern __device__ const double2* g_zrefGlob;

extern "C" void nacktmull_set_perturb(const PerturbParams& p,
                                      const double2* zrefGlobalDev) noexcept;
