///// Otter: Device-only guard log helpers; single-line ASCII; no varargs.
///// Schneefuchs: Header-only; depends on luchs_cuda_log_buffer.hpp; /WX-safe.
///  Maus: Tiny, deterministic; ints only; one final LUCHS_LOG_DEVICE(const char*).
///// Datei: src/nacktmull_devlog.cuh

#pragma once
#include <cuda_runtime.h>
#include "luchs_cuda_log_buffer.hpp"

#ifndef LUCHS_LOG_DEVICE
#define LUCHS_LOG_DEVICE(msg) do { (void)(msg); } while(0)
#endif

__device__ __forceinline__ int nm_dev_append(char* dst, const char* s){
    int n=0; while (s[n]) { dst[n]=s[n]; ++n; } return n;
}
__device__ __forceinline__ int nm_dev_itoa(char* dst, int v){
    unsigned int x = (v<0) ? (unsigned)(-v) : (unsigned)v;
    char tmp[12]; int k=0;
    do { tmp[k++] = char('0' + (x % 10)); x/=10; } while(x);
    int p=0;
    if (v<0) dst[p++]='-';
    for (int i=k-1;i>=0;--i) dst[p++]=tmp[i];
    return p;
}
__device__ __forceinline__ void nm_dev_log_guard_hit(int x,int y,int i,int len,int ver){
    char buf[96]; int p=0;
    p += nm_dev_append(buf+p, "[PERT][GUARD] xy=");
    p += nm_dev_itoa(buf+p, x);
    buf[p++]=','; p+=nm_dev_itoa(buf+p,y);
    p += nm_dev_append(buf+p, " i=");
    p += nm_dev_itoa(buf+p, i);
    p += nm_dev_append(buf+p, " len=");
    p += nm_dev_itoa(buf+p, len);
    p += nm_dev_append(buf+p, " ver=");
    p += nm_dev_itoa(buf+p, ver);
    buf[p]=0;
    LUCHS_LOG_DEVICE(buf);
}
