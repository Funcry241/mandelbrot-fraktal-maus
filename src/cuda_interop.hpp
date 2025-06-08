#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Typalias für bessere Lesbarkeit
using cudaGraphicsResource_t = struct cudaGraphicsResource*;

// ----------------------------------------------------------------------
// Kernel-Wrappers

extern "C" void launch_debugGradient(uchar4* img, int width, int height);

// 🐭 NEU: KEIN extern "C" bei C++-Signaturen mit mehr Parametern (Iterationspuffer)!
void launch_mandelbrotHybrid(
    uchar4* img,
    int* iterations,   // 🐭 Iteration Buffer
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIter
);

// ----------------------------------------------------------------------
// Gesamte CUDA-Rendering-Pipeline
namespace CudaInterop {

/// Renders a CUDA frame into a mapped OpenGL PBO
void renderCudaFrame(
    cudaGraphicsResource_t cudaPboRes,
    int width,
    int height,
    float& zoom,
    float2& offset,
    int maxIter,
    float* d_complexity,
    std::vector<float>& h_complexity,
    int* d_iterations,
    bool autoZoomEnabled    
);

}
