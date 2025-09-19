///// Otter: Split – Farbraum & Pack nach .cuh; Kernpfad/Neon-Intro/Warze & Launch bleiben hier.
//// Schneefuchs: API unverändert; weniger Zeilen; klare Abhängigkeiten; ASCII-Logs; /WX-fest.
//// Maus: Innen dunkel, außen Palette + Highlights; performantes Packen; minimale Zweige.
//// Datei: src/nacktmull.cu

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
#include "nacktmull_math.cuh"   // pixelToComplex(...)
#include "nacktmull_color.cuh"  // Farb- & Pack-Helfer (header-only)

// ============================================================================
// Animation-Uniforms (vom Host pro Frame gesetzt)
// ============================================================================
__constant__ float g_sinA = 0.0f;  // ~sin(0.30*t)
__constant__ float g_sinB = 0.0f;  // ~sin(0.80*t)

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
// Palette (kompakt) + Neon-Intro-Boost (~2s) – nutzt Farb-Helfer aus .cuh
// ============================================================================
__device__ __forceinline__ float3 gtPalette_srgb(float x, bool inSet, float t){
    const float gamma=0.84f, lift=0.08f, baseVibr=1.05f, addVibrMax=0.06f, warmDriftAmp=0.06f;
    const float breathAmp = 0.08f;

    // Neon-Intro (0..~2s): Punch ↑, dann weich zurück
    const float introT = clamp01(t * 0.5f); // t/2s -> 0..1
    const float gammaA      = mixf(0.72f, gamma,      introT);
    const float liftA       = mixf(lift + 0.12f, lift, introT);
    const float baseVibrA   = mixf(baseVibr * 1.22f,  baseVibr,  introT);
    const float addVibrMaxA = mixf(addVibrMax * 1.35f,addVibrMax,introT);
    const float warmShiftA  = mixf(1.08f, (1.00f + warmDriftAmp*g_sinA), introT);

    if (inSet) return make_float3(0.f,0.f,0.f); // Set-Innen dunkel

    x = clamp01(__powf(clamp01(x), gammaA));
    x = clamp01((x + liftA) / (1.0f + liftA));
    const float xprime = clamp01(x + breathAmp*g_sinB*x*(1.0f - x));

    // 8-Knoten-Farbspur
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
    int j = 0;
    #pragma unroll
    for (int i=0; i<7; ++i) { if (xprime >= p[i]) j = i; }
    const float span = fmaxf(p[j+1] - p[j], 1e-6f);
    float tseg = clamp01((xprime - p[j]) / span);
    tseg = tseg*tseg*(3.f-2.f*tseg); // smoothstep

    float3 a=cLn[j], b=cLn[j+1];
    float3 rgb = make_float3(a.x + (b.x-a.x)*tseg, a.y + (b.y-a.y)*tseg, a.z + (b.z-a.z)*tseg);

    // Vibrance + warme Verschiebung um Luma herum
    const float luma = zk_luma_srgb(rgb);
    const float vibr = baseVibrA + addVibrMaxA*clamp01((xprime-0.10f)*(1.0f/0.40f));
    rgb = zk_vibrance_warm_shift(rgb, luma, vibr, warmShiftA);
    return rgb;
}

// ============================================================================
// Rüsselwarze: Glanz & Glitzer (analytische Gradienten, performant)
// ============================================================================
__device__ __forceinline__ float3 warze_highlight(float2 c, int px, int py, float t, float maskGain){
    const float kR=9.0f, kA=5.0f;
    const float wob = 0.9f*g_sinA + 0.5f*g_sinB;

    // Polarkoordinaten + Phasen
    const float r2 = fmaxf(c.x*c.x + c.y*c.y, 1e-20f);
    const float r  = sqrtf(r2);
    const float ang= atan2f(c.y, c.x);
    const float phi0 = kR*r + kA*ang + wob;
    const float phiR = 0.5f*kR*r - 1.2f*kA*ang + 1.7f*wob;

    float s0, c0; __sincosf(phi0, &s0, &c0);
    const float rid = fabsf(__sinf(phiR));
    float m = clamp01(0.60f*rid + 0.40f*(s0*s0)) * maskGain;

    // Analytische Gradienten -> Pseudo-Normal von sin(phi0)
    const float invr  = rsqrtf(r2), invr2 = invr*invr;
    const float dphidx = kR*(c.x*invr) - kA*(c.y*invr2);
    const float dphidy = kR*(c.y*invr) + kA*(c.x*invr2);
    float2 n = make_float2(-c0*dphidx, -c0*dphidy);
    float invn = rsqrtf(fmaxf(n.x*n.x+n.y*n.y, 1e-18f));
    n.x *= invn; n.y *= invn;

    // Licht + Spekular
    const float2 L = make_float2(__cosf(0.21f*t), __sinf(0.17f*t));
    float spec = __powf(fmaxf(0.0f, n.x*L.x + n.y*L.y), 22.0f) * m;

    // Glitzer: pixelgesät & zeitlich moduliert
    float seed = __sinf((px*12.9898f + py*78.233f) + 6.2831853f*fractf(0.123f*t));
    float glint = __powf(fmaxf(0.0f, seed) * fmaxf(0.0f, __sinf(5.0f*s0 + 2.0f*wob)), 8.0f) * m;
    float tw = 0.5f + 0.5f*__sinf(2.4f*t + 0.8f*px + 1.1f*py);

    float glow = glint * tw;
    float3 h = make_float3(spec + glow, spec + glow*0.92f, spec + glow*0.85f);
    h.x *= 0.85f; h.y *= 0.85f; h.z *= 0.85f; // leichte Gesamtdämpfung
    return h;
}

// ============================================================================
// Basisshading (ohne Alpha / ohne Compose)
// ============================================================================
__device__ __forceinline__ float3 shade_color_only(
    int it,int maxIter,float zx,float zy,float t)
{
    if (it>=maxIter){
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
// Progressive-Status (__constant__) + Setter (API unverändert)
// ============================================================================
struct NacktmullProgState { float2* z; uint16_t* it; int addIter; int iterCap; int enabled; };
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
    const float esc2=4.0f;

    // Progressive Pfad
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

        // Basispalette
        float3 rgb = shade_color_only(it, iterCap, zx, zy, tSec);

        // Rüsselwarze nur außerhalb (it < iterCap)
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

    // [ZK] Direct Pfad (mit Periodizität) + Warp-Frühstopp (Chunk + Vote)
    float zx=0.f, zy=0.f;
    int it = maxIter; // default: bounded

    float px=0.f, py=0.f; int lastProbe=0;
    const int   perN  = Settings::periodicityCheckInterval;
    const float eps2  = (float)Settings::periodicityEps2;

    #ifndef WARP_CHUNK
    #define WARP_CHUNK 8
    #endif

    unsigned mask = __activemask();
    bool done = false;
    int i = 0;
    for (; i < maxIter; ) {
        #pragma unroll
        for (int j = 0; j < WARP_CHUNK && i < maxIter; ++j, ++i) {
            const float x2 = zx*zx, y2 = zy*zy;
            if (x2 + y2 > esc2) { it = i; done = true; break; }
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

        // Anim-Uniforms (Fehler nur loggen)
        {
            const float sinA = sinf(0.30f * tSec);
            const float sinB = sinf(0.80f * tSec);
            cudaError_t e1 = cudaMemcpyToSymbol(g_sinA, &sinA, sizeof(float));
            cudaError_t e2 = cudaMemcpyToSymbol(g_sinB, &sinB, sizeof(float));
            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                LUCHS_LOG_HOST("[NACKTMULL][WARN] memcpyToSymbol failed: a=%d b=%d",(int)e1,(int)e2);
            }
        }

        const dim3 block(Settings::MANDEL_BLOCK_X, Settings::MANDEL_BLOCK_Y);
        const dim3 grid((w+block.x-1)/block.x,(h+block.y-1)/block.y);

        cudaStream_t useStream = stream; // darf nullptr sein -> äquivalent zu Stream 0 beim Launch unten

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
                LUCHS_LOG_HOST("[PERF] nacktmull unified kern=%.2f ms itMax=%d bx=%d by=%d unroll=%d chunk=%d",
                               ms, maxIter, (int)block.x, (int)block.y, (int)Settings::MANDEL_UNROLL, (int)WARP_CHUNK);
            }
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);
        } else {
            mandelbrotUnifiedKernel<<<grid,block,0,useStream>>>(out,d_it,w,h,zoom,offset,maxIter,tSec);
        }

        // Periodizitäts-Info nur einmal loggen
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
