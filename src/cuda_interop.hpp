#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using cudaGraphicsResource_t = struct cudaGraphicsResource*;

extern "C" void launch_debugGradient(uchar4* img, int w, int h);
void launch_mandelbrotHybrid(uchar4* img, int* iters, int w, int h, float zoom, float2 offset, int maxIter);

namespace CudaInterop {
void renderCudaFrame(cudaGraphicsResource_t pboRes, int w, int h, float& zoom, float2& offset,
                     int maxIter, float* d_comp, std::vector<float>& h_comp, int* d_iters);
}
