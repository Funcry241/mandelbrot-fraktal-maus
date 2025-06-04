#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Typalias für bessere Lesbarkeit
using cudaGraphicsResource_t = struct cudaGraphicsResource*;

// Kernel-Wrappers
extern "C" void launch_debugGradient(uchar4* img, int w, int h);
extern "C" void launch_mandelbrotHybrid(uchar4* img, int w, int h, float zoom, float2 offset, int maxIter);

// Gesamte CUDA-Rendering-Pipeline
namespace CudaInterop {
void renderCudaFrame(
    cudaGraphicsResource_t cudaPboRes,
    int width,
    int height,
    float& zoom,
    float2& offset,
    int maxIter,
    float* d_complexity,
    std::vector<float>& h_complexity
);
}
