///// Otter: Defines g_pert + g_zrefGlob; implements nacktmull_set_perturb.
///  Schneefuchs: Tiny TU; avoids bloating kernel TU; /WX-safe.
///  Maus: Pointer is device-global; set via cudaMemcpyToSymbol.
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
