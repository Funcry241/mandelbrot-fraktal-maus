///// Otter: Defines g_prog + implements nacktmull_set_progressive.
///  Schneefuchs: Separate TU to cut size; API unchanged; /WX-safe.
///  Maus: Minimal includes; deterministic logging on failure only.
///// Datei: src/nacktmull_prog.cu

#include <cuda_runtime.h>
#include <vector_types.h>
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "nacktmull_prog.cuh"

__device__ __constant__ NacktmullProgState g_prog = { nullptr,nullptr,0,0,0 };

extern "C" void nacktmull_set_progressive(const void* zDev,const void* itDev,
                                          int addIter,int iterCap,int enabled) noexcept
{
    NacktmullProgState h{};
    h.z=(float2*)zDev; h.it=(uint16_t*)itDev; h.addIter=addIter; h.iterCap=iterCap; h.enabled=enabled?1:0;
    cudaError_t err = cudaMemcpyToSymbol(g_prog,&h,sizeof(h));
    if constexpr (Settings::debugLogging) {
        if (err != cudaSuccess) {
            LUCHS_LOG_HOST("[NACKTMULL][WARN] memcpyToSymbol(g_prog) failed: err=%d", (int)err);
        }
    }
}
