///// Otter: Split – Kernel & Launch only; shading + devlog + prog/pert moved.
///  Schneefuchs: API unverändert; kleiner TU; klare Abhängigkeiten; /WX-fest.
///  Maus: Innen dunkel, außen Palette + Highlights; minimale Zweige.
///// Datei: src/nacktmull.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
#include <chrono>
#include <cstdint>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"
#include "nacktmull_math.cuh"     // pixelToComplex(...)
#include "nacktmull_shade.cuh"    // gtPalette_srgb, warze_highlight, shade_color_only
#include "nacktmull_devlog.cuh"   // nm_dev_log_guard_hit(...)
#include "nacktmull_prog.cuh"     // g_prog + setter (in eigener TU)
#include "nacktmull_pert.cuh"     // g_pert/g_zrefGlob + setter
#include "core_kernel.h"          // zrefConst, PerturbParams, PertStore

// --- DE-Soft-Edge Defaults (falls nicht in Settings.hpp definiert) ----------
#ifndef DE_SOFT_EDGE_ENABLE
#define DE_SOFT_EDGE_ENABLE 1
#endif
#ifndef DE_SOFT_K
#define DE_SOFT_K 0.6f
#endif

// ============================================================================
// Animation Uniforms (definiert hier; von Launch pro Frame gesetzt)
// ============================================================================
__constant__ float g_sinA = 0.0f;  // ~sin(0.30*t)
__constant__ float g_sinB = 0.0f;  // ~sin(0.80*t)

// Telemetrie aus anderer TU
extern __device__ float d_deltaMax; // defined in core_kernel.cu

// ============================================================================
// Early-out: Haupt-Kardioide + Period-2-Knolle
// ============================================================================
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y*y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true;       // Haupt-Kardioide
    const float xp = x + 1.0f;
    if (xp*xp + y2 <= 0.0625f) return true;        // Period-2-Bulb
    return false;
}

// ============================================================================
// Unified Kernel – Direct ODER Progressive (branch by g_prog.enabled)
// ============================================================================
__global__ __launch_bounds__(Settings::MANDEL_BLOCK_X * Settings::MANDEL_BLOCK_Y)
void mandelbrotUnifiedKernel(
    uchar4* __restrict__ out, uint16_t* __restrict__ iterOut,
    int w,int h,float zoom,float2 center,int maxIter,float tSec)
{
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    const int idx=y*w+x;

    const float2 c = pixelToComplex((double)x+0.5,(double)y+0.5,w,h,(double)center.x,(double)center.y,(double)zoom);

    // Innen: dunkel & solide
    if (insideMainCardioidOrBulb(c.x,c.y)){
        const float3 rgb = gtPalette_srgb(0.0f,true,tSec);
        out[idx] = zk_pack_srgb8(rgb, x, y);
        iterOut[idx] = (uint16_t)min(maxIter, 65535);
        if (g_prog.enabled && g_prog.it && g_prog.z){ g_prog.it[idx]=maxIter; g_prog.z[idx]=make_float2(0,0); }
        return;
    }

    const bool prog = (g_prog.enabled && g_prog.z && g_prog.it && g_prog.addIter>0);
    const bool pert_on = (g_pert.active != 0) && (g_pert.len > 0);
    const float esc2=4.0f;

    // Progressive Pfad – (aktuell klassisch; Warmstart folgt in nächstem Schritt)
    if (prog){
        int it = (int)g_prog.it[idx];
        float2 z0 = g_prog.z[idx];
        float zx = (it>0) ? z0.x : 0.f;
        float zy = (it>0) ? z0.y : 0.f;

        const int iterCap = g_prog.iterCap>0 ? g_prog.iterCap : maxIter;
        int stepLimit = iterCap - it;
        if (g_prog.addIter < stepLimit) stepLimit = g_prog.addIter;
        if (stepLimit < 0) stepLimit = 0;

        #pragma unroll 1
        for (int s=0;s<stepLimit;++s){
            const float x2=zx*zx, y2=zy*zy;
            if (x2+y2>esc2){ break; }
            const float xt=x2-y2+c.x; zy=__fmaf_rn(2.0f*zx,zy,c.y); zx=xt; ++it;
        }
        g_prog.it[idx]=(uint16_t)min(it,65535); g_prog.z[idx]=make_float2(zx,zy);

        float3 rgb = shade_color_only(it, iterCap, zx, zy, tSec);
        if (it < iterCap){
            float luma = zk_luma_srgb(rgb);
            float r2=zx*zx+zy*zy; float smoothX;
            if (r2>1.0000001f && it>0){
                float r=sqrtf(r2), l2=__log2f(__log2f(r));
                smoothX = clamp01(((float)it - l2) / (float)iterCap);
            } else {
                smoothX = clamp01((float)it / (float)iterCap);
            }
            float3 add = warze_highlight(c, x, y, tSec, (0.35f + 0.65f*luma) * (0.55f + 0.45f*smoothX));
            rgb.x = clamp01(rgb.x + add.x);
            rgb.y = clamp01(rgb.y + add.y);
            rgb.z = clamp01(rgb.z + add.z);
        }

        out[idx] = zk_pack_srgb8(rgb, x, y);
        iterOut[idx]=(uint16_t)min(it,65535);
        return;
    }

    // ========================================================================
    // PERT ACTIVE PATH (klassik bleibt Fallback & Referenz)
    // ========================================================================
    double dmax = 0.0;
    if (pert_on) {
        double2 delta = make_double2(0.0, 0.0);
        double2 wder  = make_double2(0.0, 0.0); // w = 2*z*w + 1
        int it = 0;
        bool fallback = false;
        float zx_out = 0.f, zy_out = 0.f;

        for (; it < maxIter; ++it) {
            if (it >= g_pert.len) { fallback = true; break; }

            double2 zr;
            if (g_pert.store == PertStore::Const) {
                zr = zrefConst[it];
            } else {
                if (!g_zrefGlob) { fallback = true; break; }
                zr = g_zrefGlob[it];
            }

            const double2 delta_sq = make_double2(delta.x*delta.x - delta.y*delta.y,
                                                  2.0*delta.x*delta.y);
            const double2 two_zr_delta = make_double2(2.0*zr.x*delta.x - 2.0*zr.y*delta.y,
                                                      2.0*zr.x*delta.y + 2.0*zr.y*delta.x);
            const double2 c_minus_ref = make_double2((double)c.x - g_pert.c_ref.x,
                                                     (double)c.y - g_pert.c_ref.y);
            delta.x = two_zr_delta.x + delta_sq.x + c_minus_ref.x;
            delta.y = two_zr_delta.y + delta_sq.y + c_minus_ref.y;

            const double mag2 = delta.x*delta.x + delta.y*delta.y;
            const double mag  = sqrt(mag2);
            if (mag > dmax) dmax = mag;
            if (!isfinite(mag2) || mag > g_pert.deltaGuard) {
                if constexpr (Settings::debugLogging) {
                    if ((it % Settings::pertDevLogEvery) == 0) {
                        nm_dev_log_guard_hit(x, y, it, g_pert.len, g_pert.version);
                    }
                }
                fallback = true;
                break;
            }

            const double zx_d = zr.x + delta.x;
            const double zy_d = zr.y + delta.y;

            const double2 wnew = make_double2(2.0*zx_d*wder.x - 2.0*zy_d*wder.y + 1.0,
                                              2.0*zx_d*wder.y + 2.0*zy_d*wder.x);
            wder = wnew;

            const double r2 = zx_d*zx_d + zy_d*zy_d;
            if (r2 > 4.0) {
                zx_out = (float)zx_d; zy_out = (float)zy_d;
                ++it;
                break;
            }
        }

        // Block-Reduce -> d_deltaMax (float bits)
        {
            __shared__ unsigned int smax_bits2;
            if (threadIdx.x==0 && threadIdx.y==0) smax_bits2 = 0u;
            __syncthreads();
            atomicMax(&smax_bits2, __float_as_uint((float)dmax));
            __syncthreads();
            if (threadIdx.x==0 && threadIdx.y==0) {
                atomicMax((unsigned int*)&d_deltaMax, smax_bits2);
            }
        }

        if (!fallback) {
            float3 rgb = shade_color_only(it, maxIter, zx_out, zy_out, tSec);

            if (it < maxIter){
                float ampBase;
                {
                    float luma = zk_luma_srgb(rgb);
                    float r2f=zx_out*zx_out+zy_out*zy_out; float smoothX;
                    if (r2f>1.0000001f && it>0){
                        float r_=sqrtf(r2f), l2=__log2f(__log2f(r_));
                        smoothX = clamp01(((float)it - l2) / (float)maxIter);
                    } else {
                        smoothX = clamp01((float)it / (float)maxIter);
                    }
                    ampBase = (0.35f + 0.65f*luma) * (0.55f + 0.45f*smoothX);
                }

                float amp = ampBase;
#if DE_SOFT_EDGE_ENABLE
                {
                    float r = sqrtf(zx_out*zx_out + zy_out*zy_out);
                    float wabs = (float)sqrt(wder.x*wder.x + wder.y*wder.y);
                    wabs = fmaxf(wabs, 1e-18f);
                    float de = 0.5f * logf(fmaxf(r, 1e-12f)) * (r / wabs);
                    float soft = clamp01(1.0f - (float)DE_SOFT_K * de);
                    amp *= soft;
                }
#endif
                float3 add = warze_highlight(c, x, y, tSec, amp);
                rgb.x = clamp01(rgb.x + add.x);
                rgb.y = clamp01(rgb.y + add.y);
                rgb.z = clamp01(rgb.z + add.z);
            }

            out[idx]     = zk_pack_srgb8(rgb, x, y);
            iterOut[idx] = (uint16_t)min(it, 65535);
            return;
        }
        // else: Guard -> klassischer Pfad
    }

    // ========================================================================
    // Klassischer Pfad
    // ========================================================================
    float zx=0.f, zy=0.f;
    int it = maxIter;

    float px=0.f, py=0.f; int lastProbe=0;
    const int   perN  = Settings::periodicityCheckInterval;
    const float eps2  = (float)Settings::periodicityEps2;

    #ifndef WARP_CHUNK
    #define WARP_CHUNK 8
    #endif

    bool trackPert = pert_on;
    double2 delta = make_double2(0.0, 0.0);

    unsigned mask = __activemask();
    bool done = false;
    int i = 0;
    for (; i < maxIter; ) {
        #pragma unroll
        for (int j = 0; j < WARP_CHUNK && i < maxIter; ++j, ++i) {
            if (trackPert && i < g_pert.len) {
                double2 zr;
                if (g_pert.store == PertStore::Const) {
                    zr = zrefConst[i];
                } else {
                    if (!g_zrefGlob) { trackPert = false; }
                    else { zr = g_zrefGlob[i]; }
                }
                if (trackPert) {
                    const double2 delta_sq = make_double2(delta.x*delta.x - delta.y*delta.y,
                                                          2.0*delta.x*delta.y);
                    const double2 two_zr_delta = make_double2(2.0*zr.x*delta.x - 2.0*zr.y*delta.y,
                                                              2.0*zr.x*delta.y + 2.0*zr.y*delta.x);
                    const double2 c_minus_ref = make_double2((double)c.x - g_pert.c_ref.x,
                                                             (double)c.y - g_pert.c_ref.y);
                    delta.x = two_zr_delta.x + delta_sq.x + c_minus_ref.x;
                    delta.y = two_zr_delta.y + delta_sq.y + c_minus_ref.y;
                    const double mag2 = delta.x*delta.x + delta.y*delta.y;
                    const double mag  = sqrt(mag2);
                    if (mag > dmax) dmax = mag;
                    if (!isfinite(mag2) || mag > g_pert.deltaGuard) {
                        trackPert = false;
                    }
                }
            }

            const float x2 = zx*zx, y2 = zy*zy;
            if (x2 + y2 > 4.0f) { it = i; done = true; break; }
            const float xt = x2 - y2 + c.x; zy = __fmaf_rn(2.0f*zx, zy, c.y); zx = xt;

            if constexpr (Settings::periodicityEnabled) {
                const int step = i - lastProbe;
                if (step >= perN) {
                    const float ddx = zx - px, ddy = zy - py;
                    const float d2  = ddx*ddx + ddy*ddy;
                    if (d2 <= eps2) { it = maxIter; done = true; break; }
                    px = zx; py = zy; lastProbe = i;
                }
            }
        }
        if (__all_sync(mask, done)) break;
    }

    if (pert_on) {
        __shared__ unsigned int smax_bits;
        if (threadIdx.x==0 && threadIdx.y==0) smax_bits = 0u;
        __syncthreads();
        atomicMax(&smax_bits, __float_as_uint((float)dmax));
        __syncthreads();
        if (threadIdx.x==0 && threadIdx.y==0) {
            atomicMax((unsigned int*)&d_deltaMax, smax_bits);
        }
    }

    float3 rgb = shade_color_only(it, maxIter, zx, zy, tSec);

    if (it < maxIter){
        float luma = zk_luma_srgb(rgb);
        float r2=zx*zx+zy*zy; float smoothX;
        if (r2>1.0000001f && it>0){
            float r=sqrtf(r2), l2=__log2f(__log2f(r));
            smoothX = clamp01(((float)it - l2) / (float)maxIter);
        } else {
            smoothX = clamp01((float)it / (float)maxIter);
        }
        float3 add = warze_highlight(c, x, y, tSec, (0.35f + 0.65f*luma) * (0.55f + 0.45f*smoothX));
        rgb.x = clamp01(rgb.x + add.x);
        rgb.y = clamp01(rgb.y + add.y);
        rgb.z = clamp01(rgb.z + add.z);
    }

    out[idx] = zk_pack_srgb8(rgb, x, y);
    iterOut[idx]=(uint16_t)min(it,65535);
}

// ============================================================================
// Public API – Timing + Kernel-Launch (Stream-Variante)
// ============================================================================
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
        cudaError_t e1 = cudaMemcpyToSymbol(g_sinA, &sinA, sizeof(float));
        cudaError_t e2 = cudaMemcpyToSymbol(g_sinB, &sinB, sizeof(float));
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            LUCHS_LOG_HOST("[NACKTMULL][WARN] memcpyToSymbol failed: a=%d b=%d",(int)e1,(int)e2);
        }

        const dim3 block(Settings::MANDEL_BLOCK_X, Settings::MANDEL_BLOCK_Y);
        const dim3 grid((w+block.x-1)/block.x,(h+block.y-1)/block.y);

        cudaStream_t useStream = stream;

        if constexpr (Settings::performanceLogging) {
            cudaEvent_t evStart=nullptr, evStop=nullptr;
            if (cudaEventCreate(&evStart) != cudaSuccess) {
                LUCHS_LOG_HOST("[PERF][ERR] cudaEventCreate(evStart) failed");
                return;
            }
            if (cudaEventCreate(&evStop)  != cudaSuccess) {
                LUCHS_LOG_HOST("[PERF][ERR] cudaEventCreate(evStop) failed");
                cudaEventDestroy(evStart);
                return;
            }

            cudaEventRecord(evStart, useStream);
            mandelbrotUnifiedKernel<<<grid,block,0,useStream>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);
            cudaEventRecord(evStop, useStream);
            cudaEventSynchronize(evStop);

            float ms=0.0f;
            if (cudaEventElapsedTime(&ms, evStart, evStop) != cudaSuccess) {
                LUCHS_LOG_HOST("[PERF][WARN] cudaEventElapsedTime failed");
            } else {
                LUCHS_LOG_HOST("[PERF] nacktmull unified kern=%.2f ms itMax=%d bx=%d by=%d unroll=%d",
                               ms, maxIter, (int)block.x, (int)block.y, (int)Settings::MANDEL_UNROLL);
            }
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);
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
