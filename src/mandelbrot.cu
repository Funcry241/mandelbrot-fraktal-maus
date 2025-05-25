// mandelbrot.cu – aktualisierte Farbgebung integriert in bestehende Kernelstruktur

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "settings.hpp"
#include "mandelbrot.hpp"

// Sanfte Farbpalette mittels Sinusverlauf
__device__ uchar4 elegantColor(float t) {
    float r = 0.5f + 0.5f * sinf(6.2831f * t);
    float g = 0.5f + 0.5f * sinf(6.2831f * t + 2.094f);  // +120°
    float b = 0.5f + 0.5f * sinf(6.2831f * t + 4.188f);  // +240°
    return make_uchar4((unsigned char)(r * 255.0f),
                       (unsigned char)(g * 255.0f),
                       (unsigned char)(b * 255.0f),
                       255);
}

__global__ void kernel(uchar4* ptr, int w, int h, float zoom, float offX, float offY, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float jx = (x - w / 2.0f) * zoom + offX;
    float jy = (y - h / 2.0f) * zoom + offY;

    float zx = 0.0f;
    float zy = 0.0f;
    int iter = 0;

    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xtemp = zx * zx - zy * zy + jx;
        zy = 2.0f * zx * zy + jy;
        zx = xtemp;
        iter++;
    }

    float t = (iter < maxIter) ? ((float)iter / maxIter) : 0.0f;
    ptr[y * w + x] = elegantColor(t);
}

extern "C" void launch_kernel(uchar4* devPtr, int w, int h, float zoom, float offX, float offY, int maxIter) {
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                  (h + blockSize.y - 1) / blockSize.y);
    kernel<<<gridSize, blockSize>>>(devPtr, w, h, zoom, offX, offY, maxIter);
}