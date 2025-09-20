///// Otter: Device-Helpers (Early-out, Progressive/Klassik-Loops, PERT-Loop, Block-Reduce).
///// Schneefuchs: Header-only; keine Global-Defs; minimale Includes; Soft-Edge-Defaults falls undef.
///// Maus: Wenig Verzweigungen; wiederverwendbar; Telemetrie optional.
///// Datei: src/nacktmull_kernel_helpers.cuh

#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>

#include "settings.hpp"
#include "common.hpp"
#include "core_kernel.h"      // zrefConst, PerturbParams, PertStore
#include "nacktmull_prog.cuh" // g_prog
#include "nacktmull_pert.cuh" // g_pert, g_zrefGlob
#include "nacktmull_devlog.cuh"

// Soft-Edge Defaults (falls nicht in Settings.hpp definiert)
#ifndef DE_SOFT_EDGE_ENABLE
#define DE_SOFT_EDGE_ENABLE 1
#endif
#ifndef DE_SOFT_K
#define DE_SOFT_K 0.6f
#endif

extern __device__ float d_deltaMax; // defined in core_kernel.cu

// ---------- Early-out ----------
__device__ __forceinline__ bool nm_inside_maincardioid_or_bulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y*y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true;
    const float xp = x + 1.0f;
    if (xp*xp + y2 <= 0.0625f) return true;
    return false;
}

// ---------- Block-Reduce(|delta|_max) -> d_deltaMax ----------
__device__ __forceinline__ void nm_block_reduce_deltaMax(double dmax){
    __shared__ unsigned int smax_bits;
    if (threadIdx.x==0 && threadIdx.y==0) smax_bits = 0u;
    __syncthreads();
    atomicMax(&smax_bits, __float_as_uint((float)dmax));
    __syncthreads();
    if (threadIdx.x==0 && threadIdx.y==0) {
        atomicMax((unsigned int*)&d_deltaMax, smax_bits);
    }
}

// ---------- Ergebnisse ----------
struct NmIterResult { int it; float zx, zy; };
struct NmPertResult { bool fallback; int it; float zx, zy; double2 wder; double dmax; };

// ---------- Progressive (klassisch) ----------
__device__ __forceinline__ NmIterResult
nm_run_progressive_classic(const float2 c, int idx, int maxIter)
{
    const float esc2 = 4.0f;
    int it = (int)g_prog.it[idx];
    float2 z0 = g_prog.z[idx];
    float zx = (it>0) ? z0.x : 0.0f;
    float zy = (it>0) ? z0.y : 0.0f;

    const int iterCap = g_prog.iterCap>0 ? g_prog.iterCap : maxIter;
    int stepLimit = iterCap - it;
    if (g_prog.addIter < stepLimit) stepLimit = g_prog.addIter;
    if (stepLimit < 0) stepLimit = 0;

    #pragma unroll 1
    for (int s=0; s<stepLimit; ++s){
        const float x2=zx*zx, y2=zy*zy;
        if (x2+y2>esc2) break;
        const float xt=x2-y2+c.x; zy=__fmaf_rn(2.0f*zx,zy,c.y); zx=xt; ++it;
    }
    g_prog.it[idx]=(uint16_t)min(it,65535); g_prog.z[idx]=make_float2(zx,zy);
    return { it, zx, zy };
}

// ---------- PERT-Pfad (mit Guard & DE-Ableitung, ohne Shading) ----------
__device__ __forceinline__ NmPertResult
nm_run_pert_path(const float2 c, int maxIter)
{
    double2 delta = make_double2(0.0, 0.0);
    double2 wder  = make_double2(0.0, 0.0); // w = 2*z*w + 1
    double dmax   = 0.0;

    int it = 0;
    bool fallback = false;
    float zx_out = 0.f, zy_out = 0.f;

    for (; it < maxIter; ++it) {
        if (it >= g_pert.len) { fallback = true; break; }

        // z_ref laden
        double2 zr;
        if (g_pert.store == PertStore::Const) zr = zrefConst[it];
        else {
            if (!g_zrefGlob) { fallback = true; break; }
            zr = g_zrefGlob[it];
        }

        // delta_{n+1} = 2*z_ref*delta + delta^2 + (c - c_ref)
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
                    nm_dev_log_guard_hit(blockIdx.x*blockDim.x+threadIdx.x,
                                         blockIdx.y*blockDim.y+threadIdx.y,
                                         it, g_pert.len, g_pert.version);
                }
            }
            fallback = true;
            break;
        }

        // z = z_ref + delta
        const double zx_d = zr.x + delta.x;
        const double zy_d = zr.y + delta.y;

        // w = 2*z*w + 1
        const double2 wnew = make_double2(2.0*zx_d*wder.x - 2.0*zy_d*wder.y + 1.0,
                                          2.0*zx_d*wder.y + 2.0*zy_d*wder.x);
        wder = wnew;

        // Bailout
        const double r2 = zx_d*zx_d + zy_d*zy_d;
        if (r2 > 4.0) {
            zx_out = (float)zx_d; zy_out = (float)zy_d;
            ++it; // smooth escape notion
            break;
        }
    }
    return { fallback, it, zx_out, zy_out, wder, dmax };
}

// ---------- Klassischer Pfad (inkl. PERT-Telemetrie, ohne Shading) ----------
#ifndef WARP_CHUNK
#define WARP_CHUNK 8
#endif

__device__ __forceinline__ NmIterResult
nm_run_classic_path(const float2 c, int maxIter, bool pert_on, double& dmax_inout)
{
    float zx=0.f, zy=0.f;
    int it = maxIter; // bounded default

    float px=0.f, py=0.f; int lastProbe=0;
    const int   perN  = Settings::periodicityCheckInterval;
    const float eps2  = (float)Settings::periodicityEps2;

    bool trackPert = pert_on;
    double2 delta = make_double2(0.0, 0.0);

    unsigned mask = __activemask();
    bool done = false;
    int i = 0;
    for (; i < maxIter; ) {
        #pragma unroll
        for (int j = 0; j < WARP_CHUNK && i < maxIter; ++j, ++i) {
            // Telemetrie: PERT-delta
            if (trackPert && i < g_pert.len) {
                double2 zr;
                if (g_pert.store == PertStore::Const) zr = zrefConst[i];
                else {
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
                    if (mag > dmax_inout) dmax_inout = mag;
                    if (!isfinite(mag2) || mag > g_pert.deltaGuard) {
                        trackPert = false;
                    }
                }
            }

            // klassische Iteration
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
    return { it, zx, zy };
}
