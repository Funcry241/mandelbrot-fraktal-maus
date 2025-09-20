///// Otter: Deklariert g_pert/g_zrefGlob (PERT) + Setter-API.
///// Schneefuchs: Header-only Deklarationen; klare Zuständigkeiten; stabil.
///// Maus: Minimale Oberfläche; keine Abhängigkeiten außer core_kernel.h.
///// Datei: src/nacktmull_pert.cuh

#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include "core_kernel.h" // PerturbParams, PertStore

extern __device__ __constant__ PerturbParams g_pert;
extern __device__ const double2* g_zrefGlob;

extern "C" void nacktmull_set_perturb(const PerturbParams& p,
                                      const double2* zrefGlobalDev) noexcept;
