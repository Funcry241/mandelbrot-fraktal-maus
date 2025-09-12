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
#include <cstdint>

#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"
#include "nacktmull_math.cuh"  // pixelToComplex(...)

__constant__ float g_sinA = 0.0f;  // sin(0.30*t)
__constant__ float g_sinB = 0.0f;  // sin(0.80*t)

__device__ __constant__ unsigned char DITHER_BAYER8[64] = {
     0, 48, 12, 60,  3, 51, 15, 63,
    32, 16, 44, 28, 35, 19, 47, 31,
     8, 56,  4, 52, 11, 59,  7, 55,
    40, 24, 36, 20, 43, 27, 39, 23,
     2, 50, 14, 62,  1, 49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29,
    10, 58,  6, 54,  9, 57,  5, 53,
    42, 26, 38, 22, 41, 25, 37, 21
};
__device__ __forceinline__ float bayer8x8_dither(int x, int y){
    const int idx = ((y & 7) << 3) | (x & 7);
    // returns offset in [-0.5, +0.5]
    return (float(DITHER_BAYER8[idx]) * (1.0f/64.0f)) - 0.5f;
}
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
    return (c <= 0.04045f) ? (c/12.92f) : __powf((c + 0.055f)/1.055f, 2.4f);
}
__device__ __forceinline__ float  linear_to_srgb(float c){
    return (c <= 0.0031308f) ? (12.92f*c) : (1.055f*__powf(c, 1.0f/2.4f) - 0.055f);
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
    const float warmShift = 1.00f + warmDriftAmp*g_sinA;
    const float breathAmp = 0.08f;
    if (inSet) return make_float3(12/255.f,14/255.f,20/255.f); // solides dunkles Set

    x = clamp01(__powf(clamp01(x), gamma));
    x = clamp01((x + lift) / (1.0f + lift));
    const float xprime = clamp01(x + breathAmp*g_sinB*x*(1.0f - x));

    const float  p[8] = {0.00f,0.10f,0.22f,0.38f,0.55f,0.72f,0.88f,1.00f};
    const float3 cLn[8] = {
        make_float3(0.0033465f,0.0043914f,0.0103298f),
        make_float3(0.0103298f,0.0241576f,0.1589608f),
        make_float3(0.0129830f,0.3813260f,0.6375969f),
        make_float3(0.1980693f,0.7758222f,0.5457245f),
        make_float3(1.0000000f,0.7454042f,0.2541521f),
        make_float3(0.9301109f,0.3762621f,0.0423114f),
        make_float3(0.5775804f,0.0648033f,0.1946178f),
        make_float3(0.9559734f,0.9473065f,0.9215819f)
    };
    int j=0;
    #pragma unroll
    for (int i=0;i<7;++i){ if (xprime>=p[i]) j=i; }
    const float span=fmaxf(p[j+1]-p[j],1e-6f);
    float tseg=clamp01((xprime-p[j])/span);
    tseg=tseg*tseg*(3.f-2.f*tseg);

    float3 aLn=cLn[j], bLn=cLn[j+1];
    float3 rgbLn=mix3(aLn,bLn,tseg);

    const float luma=0.2126f*rgbLn.x+0.7152f*rgbLn.y+0.0722f*rgbLn.z;
    const float vibr=baseVibr + addVibrMax*clamp01((xprime-0.10f)*(1.0f/0.40f));
    rgbLn=make_float3(
        luma+(rgbLn.x-luma)*vibr*warmShift,
        luma+(rgbLn.y-luma)*vibr*1.00f,
        luma+(rgbLn.z-luma)*vibr*(2.0f-warmShift)
    );
    return make_float3(clamp01(rgbLn.x),clamp01(rgbLn.y),clamp01(rgbLn.z));
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
struct NacktmullProgState { float2* z; uint16_t* it; int addIter; int iterCap; int enabled; };
__device__ __constant__ NacktmullProgState g_prog = { nullptr,nullptr,0,0,0 };

extern "C" void nacktmull_set_progressive(const void* zDev,const void* itDev,
                                          int addIter,int iterCap,int enabled)
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

// ============================================================================
// Unified kernel: direct OR progressive (branch by g_prog.enabled)
// ============================================================================
__global__ __launch_bounds__(256)
void mandelbrotUnifiedKernel(
    uchar4* __restrict__ out, uint16_t* __restrict__ iterOut,
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
        {
        const float d = bayer8x8_dither(x,y);
        float r = fmaf(255.0f, clamp01(rgb.x), 0.5f + d);
        float g = fmaf(255.0f, clamp01(rgb.y), 0.5f + d);
        float b = fmaf(255.0f, clamp01(rgb.z), 0.5f + d);
        r = fminf(255.0f, fmaxf(0.0f, r));
        g = fminf(255.0f, fmaxf(0.0f, g));
        b = fminf(255.0f, fmaxf(0.0f, b));
        out[idx] = make_uchar4((unsigned char)r,(unsigned char)g,(unsigned char)b,255);
    }
        iterOut[idx] = (uint16_t)min(maxIter, 65535);
        if (g_prog.enabled && g_prog.it && g_prog.z){ g_prog.it[idx]=maxIter; g_prog.z[idx]=make_float2(0,0); }
        return;
    }

    const bool prog = (g_prog.enabled && g_prog.z && g_prog.it && g_prog.addIter>0);
    const float esc2=4.0f;

    if (prog){
        // Progressive path (resume)
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

        const float3 rgb = shade_color_only(it, iterCap, zx, zy, tSec);
        {
        const float d = bayer8x8_dither(x,y);
        float r = fmaf(255.0f, clamp01(rgb.x), 0.5f + d);
        float g = fmaf(255.0f, clamp01(rgb.y), 0.5f + d);
        float b = fmaf(255.0f, clamp01(rgb.z), 0.5f + d);
        r = fminf(255.0f, fmaxf(0.0f, r));
        g = fminf(255.0f, fmaxf(0.0f, g));
        b = fminf(255.0f, fmaxf(0.0f, b));
        out[idx] = make_uchar4((unsigned char)r,(unsigned char)g,(unsigned char)b,255);
    }
        iterOut[idx]=(uint16_t)min(it,65535);
        return;
    }

    // ----------------------- Direct path (with periodicity) -------------------
    float zx=0.f, zy=0.f;
    int it = maxIter; // default: bounded
    float px=0.f, py=0.f; int lastProbe=0;
    const int perN  = Settings::periodicityCheckInterval;
    const float eps2= (float)Settings::periodicityEps2;

    #pragma unroll 1
    for (int i=0;i<maxIter;++i){
        const float x2=zx*zx, y2=zy*zy;
        if (x2+y2>esc2){ it=i; break; }
        const float xt=x2-y2+c.x; zy=__fmaf_rn(2.0f*zx,zy,c.y); zx=xt;
        if constexpr (Settings::periodicityEnabled){
            const int step=i-lastProbe;
            if (step>=perN){
                const float ddx=zx-px, ddy=zy-py;
                const float d2=ddx*ddx+ddy*ddy;
                if (d2<=eps2){ it=maxIter; break; }
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
    iterOut[idx]=(uint16_t)min(it,65535);
}

// ============================================================================
// Public API (unchanged) – ms timing + unified kernel
// ============================================================================
extern "C" void launch_mandelbrotHybrid(
    uchar4* out,uint16_t* d_it,int w,int h,float zoom,float2 offset,int maxIter,int /*tile*/)
{
    using clk=std::chrono::high_resolution_clock;
    static clk::time_point anim0; static bool anim_init=false;
    if(!anim_init){ anim0=clk::now(); anim_init=true; }
    const float tSec=(float)std::chrono::duration<double>(clk::now()-anim0).count();
    const float sinA = sinf(0.30f * tSec);
    const float sinB = sinf(0.80f * tSec);
    cudaMemcpyToSymbol(g_sinA, &sinA, sizeof(float));
    cudaMemcpyToSymbol(g_sinB, &sinB, sizeof(float));
    
    const auto tStart=clk::now();

    if(!out||!d_it||w<=0||h<=0||maxIter<=0){
        LUCHS_LOG_HOST("[NACKTMULL][ERR] invalid args out=%p it=%p w=%d h=%d itMax=%d",(void*)out,(void*)d_it,w,h,maxIter);
        return;
    }

    const dim3 block(32,8);
    const dim3 grid((w+block.x-1)/block.x,(h+block.y-1)/block.y);
    if constexpr (Settings::performanceLogging) {
        cudaEvent_t evStart = nullptr, evStop = nullptr;
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));
        CUDA_CHECK(cudaEventRecord(evStart, 0));
        mandelbrotUnifiedKernel<<<grid,block>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);
        CUDA_CHECK(cudaEventRecord(evStop, 0));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        LUCHS_LOG_HOST("[PERF] nacktmull unified kern=%.2f ms itMax=%d", ms, maxIter);
        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
    } else {
        mandelbrotUnifiedKernel<<<grid,block>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);
    }
    if constexpr (Settings::debugLogging){
        LUCHS_LOG_HOST("[INFO] periodicity enabled=%d N=%d eps2=%.3e",
            (int)Settings::periodicityEnabled,(int)Settings::periodicityCheckInterval,(double)Settings::periodicityEps2);
    }
}
