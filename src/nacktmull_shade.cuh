///// Otter: Palette, Warzen-Highlights, Shade-Funktionen; header-only; nutzt g_sinA/g_sinB.
///// Schneefuchs: Keine dynamischen Zustände; Soft-Edge schaltbar; Pack über color.cuh.
///// Maus: Innen dunkel; außen farbig & weich; minimale Rechenäste.
///// Datei: src/nacktmull_shade.cuh

#include <cuda_runtime.h>
#include <cmath>
#include "nacktmull_color.cuh"   // zk_luma_srgb, zk_vibrance_warm_shift, zk_pack_srgb8
#include "common.hpp"            // clamp01, mixf, fractf

extern __constant__ float g_sinA; // set by launch TU
extern __constant__ float g_sinB;

__device__ __forceinline__ float3 gtPalette_srgb(float x, bool inSet, float t){
    const float gamma=0.84f, lift=0.08f, baseVibr=1.05f, addVibrMax=0.06f, warmDriftAmp=0.06f;
    const float breathAmp = 0.08f;

    const float introT = clamp01(t * 0.5f);
    const float gammaA      = mixf(0.72f, gamma,      introT);
    const float liftA       = mixf(lift + 0.12f, lift, introT);
    const float baseVibrA   = mixf(baseVibr * 1.22f,  baseVibr,  introT);
    const float addVibrMaxA = mixf(addVibrMax * 1.35f,addVibrMax,introT);
    const float warmShiftA  = mixf(1.08f, (1.00f + warmDriftAmp*g_sinA), introT);

    if (inSet) return make_float3(0.f,0.f,0.f);

    x = clamp01(__powf(clamp01(x), gammaA));
    x = clamp01((x + liftA) / (1.0f + liftA));
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
    int j = 0;
    #pragma unroll
    for (int i=0; i<7; ++i) { if (xprime >= p[i]) j = i; }
    const float span = fmaxf(p[j+1] - p[j], 1e-6f);
    float tseg = clamp01((xprime - p[j]) / span);
    tseg = tseg*tseg*(3.f-2.f*tseg);

    float3 a=cLn[j], b=cLn[j+1];
    float3 rgb = make_float3(a.x + (b.x-a.x)*tseg, a.y + (b.y-a.y)*tseg, a.z + (b.z-a.z)*tseg);

    const float luma = zk_luma_srgb(rgb);
    const float vibr = baseVibrA + addVibrMaxA*clamp01((xprime-0.10f)*(1.0f/0.40f));
    rgb = zk_vibrance_warm_shift(rgb, luma, vibr, warmShiftA);
    return rgb;
}

__device__ __forceinline__ float3 warze_highlight(float2 c, int px, int py, float t, float maskGain){
    const float kR=9.0f, kA=5.0f;
    const float wob = 0.9f*g_sinA + 0.5f*g_sinB;

    const float r2 = fmaxf(c.x*c.x + c.y*c.y, 1e-20f);
    const float r  = sqrtf(r2);
    const float ang= atan2f(c.y, c.x);
    const float phi0 = kR*r + kA*ang + wob;
    const float phiR = 0.5f*kR*r - 1.2f*kA*ang + 1.7f*wob;

    float s0, c0; __sincosf(phi0, &s0, &c0);
    const float rid = fabsf(__sinf(phiR));
    float m = clamp01(0.60f*rid + 0.40f*(s0*s0)) * maskGain;

    const float invr  = rsqrtf(r2), invr2 = invr*invr;
    const float dphidx = kR*(c.x*invr) - kA*(c.y*invr2);
    const float dphidy = kR*(c.y*invr) + kA*(c.x*invr2);
    float2 n = make_float2(-c0*dphidx, -c0*dphidy);
    float invn = rsqrtf(fmaxf(n.x*n.x+n.y*n.y, 1e-18f));
    n.x *= invn; n.y *= invn;

    const float2 L = make_float2(__cosf(0.21f*t), __sinf(0.17f*t));
    float spec = __powf(fmaxf(0.0f, n.x*L.x + n.y*L.y), 22.0f) * m;

    float seed = __sinf((px*12.9898f + py*78.233f) + 6.2831853f*fractf(0.123f*t));
    float glint = __powf(fmaxf(0.0f, seed) * fmaxf(0.0f, __sinf(5.0f*s0 + 2.0f*wob)), 8.0f) * m;
    float tw = 0.5f + 0.5f*__sinf(2.4f*t + 0.8f*px + 1.1f*py);

    float glow = glint * tw;
    float3 h = make_float3(spec + glow, spec + glow*0.92f, spec + glow*0.85f);
    h.x *= 0.85f; h.y *= 0.85f; h.z *= 0.85f;
    return h;
}

__device__ __forceinline__ float3 shade_color_only(int it,int maxIter,float zx,float zy,float t){
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
