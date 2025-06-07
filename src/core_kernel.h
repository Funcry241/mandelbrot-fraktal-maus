#pragma once

#include <vector_types.h> // uchar4, float2

extern "C" void launch_mandelbrotHybrid(uchar4* img, int* iters, int w, int h, float zoom, float2 offset, int maxIter);
extern "C" void launch_debugGradient(uchar4* img, int w, int h);

__global__ void computeComplexity(const int* iters, int w, int h, float* comp, float threshold);
