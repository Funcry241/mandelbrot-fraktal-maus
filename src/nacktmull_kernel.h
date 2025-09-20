///// Otter: Kernel-Vorwärtsdeklaration für Launch-TU; keine schweren Includes.
///// Schneefuchs: Entkoppelt Build; nur Minimal-Typen; stabil gegen /WX.
///// Maus: Schlichte Signatur; eindeutige Parameter.
///// Datei: src/nacktmull_kernel.h

#pragma once
#include <vector_types.h>

__global__ void mandelbrotUnifiedKernel(
    uchar4* __restrict__ out, uint16_t* __restrict__ iterOut,
    int w,int h,float zoom,float2 center,int maxIter,float tSec);
