///// Otter: Kernel only (Unified); shading/devlog/prog/pert in Headers; Launch ausgelagert.
///// Schneefuchs: API unverändert; /WX-fest; klare Abhängigkeiten; deterministische Pfade; Log rate-limited.
///// Maus: Innen dunkel, außen Palette + Highlights; minimale Zweige; PERT Fallback → klassisch.
///// Datei: src/nacktmull.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cstdint>

#include "settings.hpp"
#include "common.hpp"
#include "core_kernel.h"          // zrefConst
#include "nacktmull_math.cuh"     // pixelToComplex(...)
#include "nacktmull_shade.cuh"    // palette, warze, shade_color_only (extern g_sinA/B)
#include "nacktmull_kernel_helpers.cuh"

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
    if (nm_inside_maincardioid_or_bulb(c.x,c.y)){
        const float3 rgb = gtPalette_srgb(0.0f,true,tSec);
        out[idx] = zk_pack_srgb8(rgb, x, y);
        iterOut[idx] = (uint16_t)min(maxIter, 65535);
        if (g_prog.enabled && g_prog.it && g_prog.z){ g_prog.it[idx]=maxIter; g_prog.z[idx]=make_float2(0,0); }
        return;
    }

    const bool prog    = (g_prog.enabled && g_prog.z && g_prog.it && g_prog.addIter>0);
    const bool pert_on = (g_pert.active != 0) && (g_pert.len > 0);

    // Progressive (klassisch)
    if (prog){
        NmIterResult pr = nm_run_progressive_classic(c, idx, maxIter);
        const int iterCap = (g_prog.iterCap>0?g_prog.iterCap:maxIter);
        float3 rgb = shade_color_only(pr.it, iterCap, pr.zx, pr.zy, tSec);
        if (pr.it < iterCap){
            float luma = zk_luma_srgb(rgb);
            const float r2=pr.zx*pr.zx+pr.zy*pr.zy;
            float smoothX = (r2>1.0000001f && pr.it>0)
                ? clamp01(((float)pr.it - __log2f(__log2f(sqrtf(r2)))) / (float)iterCap)
                : clamp01((float)pr.it / (float)iterCap);
            float3 add = warze_highlight(c, x, y, tSec, (0.35f + 0.65f*luma) * (0.55f + 0.45f*smoothX));
            rgb.x = clamp01(rgb.x + add.x); rgb.y = clamp01(rgb.y + add.y); rgb.z = clamp01(rgb.z + add.z);
        }
        out[idx] = zk_pack_srgb8(rgb, x, y);
        iterOut[idx]=(uint16_t)min(pr.it,65535);
        return;
    }

    // PERT-Pfad
    double dmax = 0.0;
    if (pert_on){
        NmPertResult pp = nm_run_pert_path(c, maxIter);
        dmax = (pp.dmax > dmax) ? pp.dmax : dmax;
        nm_block_reduce_deltaMax(dmax);

        if (!pp.fallback){
            float3 rgb = shade_color_only(pp.it, maxIter, pp.zx, pp.zy, tSec);
#if DE_SOFT_EDGE_ENABLE
            if (pp.it < maxIter){
                float r = sqrtf(pp.zx*pp.zx + pp.zy*pp.zy);
                float wabs = (float)sqrt(pp.wder.x*pp.wder.x + pp.wder.y*pp.wder.y);
                wabs = fmaxf(wabs, 1e-18f);
                float de = 0.5f * logf(fmaxf(r, 1e-12f)) * (r / wabs);
                float soft = clamp01(1.0f - (float)DE_SOFT_K * de);
                float luma = zk_luma_srgb(rgb);
                float r2f = pp.zx*pp.zx+pp.zy*pp.zy;
                float smoothX = (r2f>1.0000001f && pp.it>0)
                    ? clamp01(((float)pp.it - __log2f(__log2f(sqrtf(r2f)))) / (float)maxIter)
                    : clamp01((float)pp.it / (float)maxIter);
                float amp = (0.35f + 0.65f*luma) * (0.55f + 0.45f*smoothX) * soft;
                float3 add = warze_highlight(c, x, y, tSec, amp);
                rgb.x = clamp01(rgb.x + add.x); rgb.y = clamp01(rgb.y + add.y); rgb.z = clamp01(rgb.z + add.z);
            }
#endif
            out[idx] = zk_pack_srgb8(rgb, x, y);
            iterOut[idx]=(uint16_t)min(pp.it,65535);
            return;
        }
        // else -> klassischer Pfad
    }

    // Klassischer Pfad (mit optionaler PERT-Telemetrie)
    NmIterResult cr = nm_run_classic_path(c, maxIter, pert_on, dmax);
    if (pert_on) nm_block_reduce_deltaMax(dmax);

    float3 rgb = shade_color_only(cr.it, maxIter, cr.zx, cr.zy, tSec);
    if (cr.it < maxIter){
        float luma = zk_luma_srgb(rgb);
        const float r2=cr.zx*cr.zx+cr.zy*cr.zy;
        float smoothX = (r2>1.0000001f && cr.it>0)
            ? clamp01(((float)cr.it - __log2f(__log2f(sqrtf(r2)))) / (float)maxIter)
            : clamp01((float)cr.it / (float)maxIter);
        float3 add = warze_highlight(c, x, y, tSec, (0.35f + 0.65f*luma) * (0.55f + 0.45f*smoothX));
        rgb.x = clamp01(rgb.x + add.x); rgb.y = clamp01(rgb.y + add.y); rgb.z = clamp01(rgb.z + add.z);
    }
    out[idx] = zk_pack_srgb8(rgb, x, y);
    iterOut[idx]=(uint16_t)min(cr.it,65535);
}
