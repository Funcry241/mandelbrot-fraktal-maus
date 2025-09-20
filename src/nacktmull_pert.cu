///// Otter: Definiert g_pert/g_zrefGlob; implementiert nacktmull_set_perturb.
///// Schneefuchs: Eigene TU; ohne Logs; /WX-fest; schlank.
///// Maus: Minimale Includes; klare Datenpfade.
///// Datei: src/nacktmull_pert.cu

#include <cuda_runtime.h>
#include <vector_types.h>
#include "nacktmull_pert.cuh"

__device__ __constant__ PerturbParams g_pert = {0,0,0,PertStore::Const, {0.0,0.0}, 0.0, 0};
__device__ const double2* g_zrefGlob = nullptr;

extern "C" void nacktmull_set_perturb(const PerturbParams& p, const double2* zrefGlobalDev) noexcept
{
    (void)cudaMemcpyToSymbol(g_pert, &p, sizeof(PerturbParams));
    (void)cudaMemcpyToSymbol(g_zrefGlob, &zrefGlobalDev, sizeof(zrefGlobalDev));
}
