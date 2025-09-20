///// Otter: Launch-only TU; setzt Anim-Uniforms und ruft den Unified-Kernel; ASCII-Logs.
///// Schneefuchs: Nutzt übergebenen Stream; cudaEvent-Perf-Timer; robuste Fehlerpfade; kein API-Drift.
///// Maus: Klein & klar; keine globalen Zustände außer __constant__ Uniforms.
///// Datei: src/nacktmull_launch.cu

#include <cuda_runtime.h>
#include <chrono>
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "nacktmull_shade.cuh"  // extern __constant__ g_sinA/g_sinB
#include "nacktmull_kernel.h"   // mandelbrotUnifiedKernel decl

extern __constant__ float g_sinA;
extern __constant__ float g_sinB;

extern "C" void launch_mandelbrotHybrid(
    uchar4* out, uint16_t* d_it,
    int w, int h, float zoom, float2 offset,
    int maxIter, int /*tile*/,
    cudaStream_t stream) noexcept
{
    using clk = std::chrono::high_resolution_clock;
    try {
        static clk::time_point anim0; static bool anim_init=false;
        if(!anim_init){ anim0=clk::now(); anim_init=true; }
        const float tSec=(float)std::chrono::duration<double>(clk::now()-anim0).count();

        if(!out||!d_it||w<=0||h<=0||maxIter<=0){
            LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",
                           (void*)out,(void*)d_it,w,h,maxIter);
            return;
        }

        const float sinA = sinf(0.30f * tSec);
        const float sinB = sinf(0.80f * tSec);
        (void)cudaMemcpyToSymbol(g_sinA, &sinA, sizeof(float));
        (void)cudaMemcpyToSymbol(g_sinB, &sinB, sizeof(float));

        const dim3 block(Settings::MANDEL_BLOCK_X, Settings::MANDEL_BLOCK_Y);
        const dim3 grid((w+block.x-1)/block.x,(h+block.y-1)/block.y);
        cudaStream_t useStream = stream;

        if constexpr (Settings::performanceLogging) {
            cudaEvent_t evStart=nullptr, evStop=nullptr;
            if (cudaEventCreate(&evStart) != cudaSuccess ||
                cudaEventCreate(&evStop)  != cudaSuccess) {
                LUCHS_LOG_HOST("[PERF][ERR] cudaEventCreate failed");
                if (evStart) cudaEventDestroy(evStart);
                if (evStop)  cudaEventDestroy(evStop);
                return;
            }
            cudaEventRecord(evStart, useStream);
            mandelbrotUnifiedKernel<<<grid,block,0,useStream>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);
            cudaEventRecord(evStop, useStream);
            cudaEventSynchronize(evStop);
            float ms=0.0f;
            if (cudaEventElapsedTime(&ms, evStart, evStop) == cudaSuccess) {
                LUCHS_LOG_HOST("[PERF] nacktmull unified kern=%.2f ms itMax=%d bx=%d by=%d unroll=%d",
                               ms, maxIter, (int)block.x, (int)block.y, (int)Settings::MANDEL_UNROLL);
            }
            cudaEventDestroy(evStart); cudaEventDestroy(evStop);
        } else {
            mandelbrotUnifiedKernel<<<grid,block,0,useStream>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);
        }

        static bool s_logPeriodicityOnce = false;
        if constexpr (Settings::debugLogging){
            if (!s_logPeriodicityOnce) {
                LUCHS_LOG_HOST("[INFO] periodicity enabled=%d N=%d eps2=%.3e",
                    (int)Settings::periodicityEnabled,(int)Settings::periodicityCheckInterval,(double)Settings::periodicityEps2);
                s_logPeriodicityOnce = true;
            }
        }
    } catch (...) {
        LUCHS_LOG_HOST("[NACKTMULL][ERR] unexpected exception in launch_mandelbrotHybrid");
        return;
    }
}
