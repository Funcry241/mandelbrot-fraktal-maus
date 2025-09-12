///// Otter: Glas/Alpha komplett entfernt – reine Direktfarbe, A=255; Background-Shader/Compose entfallen.
///// Schneefuchs: Periodizität & Progressive bleiben; keine z'-Ableitung/DE mehr; kompakt, ASCII-only.
///  Maus: Innenfarbe solide (dunkles Set), Außen smooth via Palette; Launch-Bounds unverändert.
///// Datei: src/nacktmull.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
#include <chrono>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"
#include "nacktmull_math.cuh"  // pixelToComplex(...)

// ============================================================================
// Device utilities
// ============================================================================
__device__ __forceinline__ float clamp01(float x){ return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ __forceinline__ float mixf(float a, float b, float t){ return a + t*(b - a); }
__device__ __forceinline__ float3 mix3(const float3 a, const float3 b, float t){
    return make_float3(mixf(a.x,b.x,t), mixf(a.y,b.y,t), mixf(a.z,b.z,t));
}
// sRGB <-> Linear (für Palettenmischung im Linearraum)
__device__ __forceinline__ float  srgb_to_linear(float c){
    return (c <= 0.04045f) ? (c/12.92f) : powf((c + 0.055f)/1.055f, 2.4f);
}
__device__ __forceinline__ float  linear_to_srgb(float c){
    return (c <= 0.0031308f) ? (12.92f*c) : (1.055f*powf(c, 1.0f/2.4f) - 0.055f);
}
__device__ __forceinline__ float3 srgb_to_linear3(const float3 c){
    return make_float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}
__device__ __forceinline__ float3 linear_to_srgb3(const float3 c){
    return make_float3(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

// Early-out: Cardioid / Period-2-Bulb
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y*y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true;
    const float xp = x + 1.0f;
    if (xp*xp + y2 <= 0.0625f) return true;
    return false;
}

// ============================================================================
// GT palette (linear gemischt), Rückgabe sRGB (0..1)
// ============================================================================
__device__ __forceinline__ float3 gtPalette_srgb(float x, bool inSet, float t){
    const float gamma=0.84f, lift=0.08f, baseVibr=1.05f, addVibrMax=0.06f, warmDriftAmp=0.06f;
    const float warmShift = 1.00f + warmDriftAmp*__sinf(0.30f*t);
    const float breathAmp = 0.08f;
    if (inSet) return make_float3(12/255.f,14/255.f,20/255.f); // solides dunkles Set

    x = clamp01(powf(clamp01(x), gamma));
    x = clamp01((x + lift) / (1.0f + lift));
    const float xprime = clamp01(x + breathAmp*__sinf(0.80f*t)*x*(1.0f - x));

    const float  p[8] = {0.00f,0.10f,0.22f,0.38f,0.55f,0.72f,0.88f,1.00f};
    const float3 c[8] = {
        make_float3(11/255.f,14/255.f,26/255.f),
        make_float3(26/255.f,43/255.f,111/255.f),
        make_float3(30/255.f,166/255.f,209/255.f),
        make_float3(123/255.f,228/255.f,195/255.f),
        make_float3(255/255.f,224/255.f,138/255.f),
        make_float3(247/255.f,165/255.f,58/255.f),
        make_float3(200/255.f,72/255.f,122/255.f),
        make_float3(250/255.f,249/255.f,246/255.f)
    };
    int j=0;
    #pragma unroll
    for (int i=0;i<7;++i){ if (xprime>=p[i]) j=i; }
    const float span=fmaxf(p[j+1]-p[j],1e-6f);
    float tseg=clamp01((xprime-p[j])/span);
    tseg=tseg*tseg*(3.f-2.f*tseg);

    float3 aLn=srgb_to_linear3(c[j]), bLn=srgb_to_linear3(c[j+1]);
    float3 rgbLn=mix3(aLn,bLn,tseg);

    const float luma=0.2126f*rgbLn.x+0.7152f*rgbLn.y+0.0722f*rgbLn.z;
    const float vibr=baseVibr + addVibrMax*clamp01((xprime-0.10f)*(1.0f/0.40f));
    rgbLn=make_float3(
        luma+(rgbLn.x-luma)*vibr*warmShift,
        luma+(rgbLn.y-luma)*vibr*1.00f,
        luma+(rgbLn.z-luma)*vibr*(2.0f-warmShift)
    );
    return linear_to_srgb3(make_float3(clamp01(rgbLn.x),clamp01(rgbLn.y),clamp01(rgbLn.z)));
}

// ============================================================================
// Farbwahl (ohne Alpha / ohne Background-Compose)
// ============================================================================
__device__ __forceinline__ float3 shade_color_only(
    int it,int maxIter,float zx,float zy,float t)
{
    if (it>=maxIter){
        // Innen: solides dunkles Set
        return gtPalette_srgb(0.0f,true,t);
    } else {
        const float r2=zx*zx+zy*zy;
        if (r2>1.0000001f && it>0){
            const float r=sqrtf(r2), l2=__log2f(__log2f(r));
            float x=((float)it - l2) / (float)maxIter;
            float edge=clamp01(1.0f-0.75f*l2);
            x=clamp01(x+0.15f*edge*(1.0f-x));
            return gtPalette_srgb(x,false,t);
        } else {
            return gtPalette_srgb(clamp01((float)it/(float)maxIter),false,t);
        }
    }
}

// ============================================================================
// Progressive state (__constant__) + setter (unverändert)
// ============================================================================
struct NacktmullProgState { float2* z; int* it; int addIter; int iterCap; int enabled; };
__device__ __constant__ NacktmullProgState g_prog = { nullptr,nullptr,0,0,0 };

extern "C" void nacktmull_set_progressive(const void* zDev,const void* itDev,
                                          int addIter,int iterCap,int enabled)
{
    NacktmullProgState h{};
    h.z=(float2*)zDev; h.it=(int*)itDev; h.addIter=addIter; h.iterCap=iterCap; h.enabled=enabled?1:0;
    cudaError_t err = cudaMemcpyToSymbol(g_prog,&h,sizeof(h));
    if constexpr (Settings::debugLogging) {
        if (err != cudaSuccess) {
            LUCHS_LOG_HOST("[NACKTMULL][WARN] memcpyToSymbol(g_prog) failed: err=%d", (int)err);
        }
    }
}

// ============================================================================
// Unified kernel: direct OR progressive (branch by g_prog.enabled)
// ============================================================================
__global__ __launch_bounds__(256)
void mandelbrotUnifiedKernel(
    uchar4* __restrict__ out, int* __restrict__ iterOut,
    int w,int h,float zoom,float2 center,int maxIter,float tSec)
{
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    const int idx=y*w+x;

    const float2 c = pixelToComplex((double)x+0.5,(double)y+0.5,w,h,(double)center.x,(double)center.y,(double)zoom);

    // Interior shortcut
    if (insideMainCardioidOrBulb(c.x,c.y)){
        const float3 rgb = gtPalette_srgb(0.0f,true,tSec);
        out[idx] = make_uchar4(
            (unsigned char)(255.0f*clamp01(rgb.x)+0.5f),
            (unsigned char)(255.0f*clamp01(rgb.y)+0.5f),
            (unsigned char)(255.0f*clamp01(rgb.z)+0.5f),
            255);
        iterOut[idx] = maxIter;
        if (g_prog.enabled && g_prog.it && g_prog.z){ g_prog.it[idx]=maxIter; g_prog.z[idx]=make_float2(0,0); }
        return;
    }

    const bool prog = (g_prog.enabled && g_prog.z && g_prog.it && g_prog.addIter>0);
    const float esc2=4.0f;

    if (prog){
        // Progressive path (resume)
        int it = g_prog.it[idx];
        float2 z0 = g_prog.z[idx];
        float zx = (it>0) ? z0.x : 0.f;
        float zy = (it>0) ? z0.y : 0.f;

        const int iterCap = g_prog.iterCap>0 ? g_prog.iterCap : maxIter;
        int stepLimit = iterCap - it;
        if (g_prog.addIter < stepLimit) stepLimit = g_prog.addIter;
        if (stepLimit < 0) stepLimit = 0;

        bool escaped=false;
        #pragma unroll 1
        for (int s=0;s<stepLimit;++s){
            const float x2=zx*zx, y2=zy*zy;
            if (x2+y2>esc2){ escaped=true; break; }
            const float xt=x2-y2+c.x; zy=__fmaf_rn(2.0f*zx,zy,c.y); zx=xt; ++it;
        }
        g_prog.it[idx]=it; g_prog.z[idx]=make_float2(zx,zy);

        const float3 rgb = shade_color_only(it, iterCap, zx, zy, tSec);
        out[idx] = make_uchar4(
            (unsigned char)(255.0f*clamp01(rgb.x)+0.5f),
            (unsigned char)(255.0f*clamp01(rgb.y)+0.5f),
            (unsigned char)(255.0f*clamp01(rgb.z)+0.5f),
            255);
        iterOut[idx]=it;
        return;
    }

    // ----------------------- Direct path (with periodicity) -------------------
    float zx=0.f, zy=0.f; bool escaped=false;
    int it = maxIter; // default: bounded
    float px=0.f, py=0.f; int lastProbe=0;
    const int perN  = Settings::periodicityCheckInterval;
    const float eps2= (float)Settings::periodicityEps2;

    #pragma unroll 1
    for (int i=0;i<maxIter;++i){
        const float x2=zx*zx, y2=zy*zy;
        if (x2+y2>esc2){ it=i; escaped=true; break; }
        const float xt=x2-y2+c.x; zy=__fmaf_rn(2.0f*zx,zy,c.y); zx=xt;
        if constexpr (Settings::periodicityEnabled){
            const int step=i-lastProbe;
            if (step>=perN){
                const float ddx=zx-px, ddy=zy-py;
                const float d2=ddx*ddx+ddy*ddy;
                if (d2<=eps2){ it=maxIter; escaped=false; break; }
                px=zx; py=zy; lastProbe=i;
            }
        }
    }

    const float3 rgb = shade_color_only(it, maxIter, zx, zy, tSec);
    out[idx] = make_uchar4(
        (unsigned char)(255.0f*clamp01(rgb.x)+0.5f),
        (unsigned char)(255.0f*clamp01(rgb.y)+0.5f),
        (unsigned char)(255.0f*clamp01(rgb.z)+0.5f),
        255);
    iterOut[idx]=it;
}

// ============================================================================
// Public API (unchanged) – ms timing + unified kernel
// ============================================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out,int* d_it,int w,int h,float zoom,float2 offset,int maxIter,int /*tile*/)
{
    using clk=std::chrono::high_resolution_clock;
    static clk::time_point anim0; static bool anim_init=false;
    if(!anim_init){ anim0=clk::now(); anim_init=true; }
    const float tSec=(float)std::chrono::duration<double>(clk::now()-anim0).count();
    const auto tStart=clk::now();

    if(!out||!d_it||w<=0||h<=0||maxIter<=0){
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",(void*)out,(void*)d_it,w,h,maxIter);
        return;
    }

    const dim3 block(32,8);
    const dim3 grid((w+block.x-1)/block.x,(h+block.y-1)/block.y);
    mandelbrotUnifiedKernel<<<grid,block>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);

    if constexpr (Settings::performanceLogging){
        cudaDeviceSynchronize();
        const double ms=1e-3*(double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now()-tStart).count();
        LUCHS_LOG_HOST("[PERF] nacktmull unified kern=%.2f ms itMax=%d", ms, maxIter);
    }
    if constexpr (Settings::debugLogging){
        LUCHS_LOG_HOST("[INFO] periodicity enabled=%d N=%d eps2=%.3e",
            (int)Settings::periodicityEnabled,(int)Settings::periodicityCheckInterval,(double)Settings::periodicityEps2);
    }
}
