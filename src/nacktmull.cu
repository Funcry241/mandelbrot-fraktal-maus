///// Otter: Unified progressive+direct kernel; kompakt; kein ABI-Bruch; ASCII-only logs.
///// Schneefuchs: __constant__-State + Setter; DE-Alpha-Recompute nur bei Escape; ms-Timing korrekt.
///  Maus: Periodizität nur im Direktpfad (if constexpr); Innen/Compose gekapselt; Launch-Bounds unverändert.
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
__device__ __forceinline__ float smootherstep(float a, float b, float x){
    float t = clamp01((x - a) / fmaxf(b - a, 1e-6f));
    return t*t*(3.0f - 2.0f*t);
}
// sRGB <-> Linear
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

// Early-out set test
__device__ __forceinline__ bool insideMainCardioidOrBulb(float x, float y){
    const float x1 = x - 0.25f;
    const float y2 = y*y;
    const float q  = x1*x1 + y2;
    if (q*(q + x1) <= 0.25f*y2) return true;
    const float xp = x + 1.0f;
    if (xp*xp + y2 <= 0.0625f) return true;
    return false;
}

// Background (linear)
__device__ __forceinline__ float3 background_linear(float u, float v, float t){
    const float3 A = srgb_to_linear3(make_float3( 8/255.f, 10/255.f, 16/255.f));
    const float3 B = srgb_to_linear3(make_float3(28/255.f, 30/255.f, 44/255.f));
    float w = clamp01(0.25f + 0.75f*(0.6f*u + 0.4f*(1.0f - v)));
    float3 base = mix3(A, B, w);
    float phase  = 3.0f*u + 2.0f*v + 0.20f*t;
    float phase2 = 5.0f*u - 3.0f*v - 0.15f*t;
    float caust  = 0.5f + 0.5f*__sinf(phase) * (0.5f + 0.5f*__cosf(phase2));
    float boost  = 1.0f + 0.02f*(caust*caust - 0.25f);
    base.x *= boost; base.y *= boost; base.z *= boost;
    return make_float3(clamp01(base.x), clamp01(base.y), clamp01(base.z));
}

// GT palette
__device__ __forceinline__ float3 gtPalette_srgb(float x, bool inSet, float t){
    const float gamma=0.84f, lift=0.08f, baseVibr=1.05f, addVibrMax=0.06f, warmDriftAmp=0.06f;
    const float warmShift = 1.00f + warmDriftAmp*__sinf(0.30f*t);
    const float breathAmp = 0.08f;
    if (inSet) return make_float3(12/255.f,14/255.f,20/255.f);
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

// Shade (color + alpha)
__device__ __forceinline__ void shade_color_alpha(
    int it,int maxIter,float zx,float zy,float dx,float dy,bool escaped,float t,
    float3& out_srgb,float& out_alpha)
{
    if (it>=maxIter){
        out_srgb = gtPalette_srgb(0.0f,true,t);
    } else {
        const float r2=zx*zx+zy*zy;
        if (r2>1.0000001f && it>0){
            const float r=sqrtf(r2), l2=__log2f(__log2f(r));
            float x=((float)it - l2) / (float)maxIter;
            float edge=clamp01(1.0f-0.75f*l2);
            x=clamp01(x+0.15f*edge*(1.0f-x));
            out_srgb=gtPalette_srgb(x,false,t);
        } else {
            out_srgb=gtPalette_srgb(clamp01((float)it/(float)maxIter),false,t);
        }
    }
    if (!escaped){ out_alpha=0.18f; return; }
    const float r=fmaxf(1e-7f,sqrtf(zx*zx+zy*zy));
    const float dmag=fmaxf(1e-12f,sqrtf(dx*dx+dy*dy));
    const float DE=(r*logf(r))/dmag, invDE=1.0f/(DE+1e-6f);
    float A=smootherstep(0.02f,0.20f,invDE);
    const float l2=__log2f(__log2f(r)), F=clamp01(1.0f-0.80f*l2);
    A=clamp01(A*(0.60f+0.40f*F));
    const float breathe=0.12f*__sinf(0.60f*t);
    float transp=clamp01(1.0f-A); transp=clamp01(transp*(1.0f+breathe));
    out_alpha=clamp01(1.0f-transp);
}

// Compose helper
__device__ __forceinline__ void compose_write(
    int x,int y,int w,int h,float t,const float3& fg_srgb,float alpha, uchar4& outPix)
{
    const float u=((float)x+0.5f)/(float)w, v=((float)y+0.5f)/(float)h;
    const float3 bgLn=background_linear(u,v,t);
    const float3 fgLn=srgb_to_linear3(fg_srgb);
    float3 compLn=make_float3(
        alpha*fgLn.x+(1.0f-alpha)*bgLn.x,
        alpha*fgLn.y+(1.0f-alpha)*bgLn.y,
        alpha*fgLn.z+(1.0f-alpha)*bgLn.z);
    const float3 compS=linear_to_srgb3(compLn);
    outPix = make_uchar4(
        (unsigned char)(255.0f*clamp01(compS.x)+0.5f),
        (unsigned char)(255.0f*clamp01(compS.y)+0.5f),
        (unsigned char)(255.0f*clamp01(compS.z)+0.5f),
        255);
}

// ============================================================================
// Progressive state (__constant__) + setter
// ============================================================================
struct NacktmullProgState { float2* z; int* it; int addIter; int iterCap; int enabled; };
__device__ __constant__ NacktmullProgState g_prog = { nullptr,nullptr,0,0,0 };

extern "C" void nacktmull_set_progressive(const void* zDev,const void* itDev,
                                          int addIter,int iterCap,int enabled)
{
    NacktmullProgState h{};
    h.z=(float2*)zDev; h.it=(int*)itDev; h.addIter=addIter; h.iterCap=iterCap; h.enabled=enabled?1:0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_prog,&h,sizeof(h)));
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

    // Interior shortcut (shared)
    if (insideMainCardioidOrBulb(c.x,c.y)){
        const float3 glassS=gtPalette_srgb(0.0f,true,tSec);
        float alpha=0.18f; uchar4 pix;
        compose_write(x,y,w,h,tSec,glassS,alpha,pix);
        out[idx]=pix; iterOut[idx]=maxIter;
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

        float3 rgb; float alpha;
        if (escaped){
            // Recompute derivative up to 'it' for proper DE alpha
            float rx=0.f, ry=0.f, dx=0.f, dy=0.f;
            for (int k=0;k<it;++k){
                const float x2=rx*rx, y2=ry*ry; if (x2+y2>esc2) break;
                const float twx=2.0f*rx, twy=2.0f*ry;
                const float ndx=twx*dx - twy*dy + 1.0f;
                const float ndy=twx*dy + twy*dx;
                const float xt=x2-y2+c.x; ry=__fmaf_rn(2.0f*rx,ry,c.y); rx=xt; dx=ndx; dy=ndy;
            }
            shade_color_alpha(it, iterCap, zx, zy, dx, dy, true, tSec, rgb, alpha);
        } else {
            float ddx=0.f, ddy=0.f;
            shade_color_alpha(it, iterCap, zx, zy, ddx, ddy, false, tSec, rgb, alpha);
        }
        uchar4 pix; compose_write(x,y,w,h,tSec,rgb,alpha,pix);
        out[idx]=pix; iterOut[idx]=it;
        return;
    }

    // ----------------------- Direct path (with periodicity) -------------------
    float zx=0.f, zy=0.f, dx=0.f, dy=0.f; bool escaped=false;
    int it = maxIter; // default: bounded interior
    float px=0.f, py=0.f; int lastProbe=0;
    const int perN  = Settings::periodicityCheckInterval;
    const float eps2= (float)Settings::periodicityEps2;

    #pragma unroll 1
    for (int i=0;i<maxIter;++i){
        const float x2=zx*zx, y2=zy*zy;
        if (x2+y2>esc2){ it=i; escaped=true; break; }
        const float twx=2.0f*zx, twy=2.0f*zy;
        const float ndx=twx*dx - twy*dy + 1.0f;
        const float ndy=twx*dy + twy*dx;
        const float xt=x2-y2+c.x; zy=__fmaf_rn(2.0f*zx,zy,c.y); zx=xt;
        dx=ndx; dy=ndy;
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

    float3 rgb; float alpha;
    shade_color_alpha(it, maxIter, zx, zy, dx, dy, escaped, tSec, rgb, alpha);
    uchar4 pix; compose_write(x,y,w,h,tSec,rgb,alpha,pix);
    out[idx]=pix; iterOut[idx]=it;
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
