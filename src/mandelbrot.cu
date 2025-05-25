// mandelbrot.cu – neue Farbgebung: smooth & elegant

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "settings.hpp"
#include "mandelbrot.hpp"

__device__ uchar4 elegantColor(float t) {
    float r = 0.5f + 0.5f * sinf(6.2831f * t);
    float g = 0.5f + 0.5f * sinf(6.2831f * t + 2.094f);
    float b = 0.5f + 0.5f * sinf(6.2831f * t + 4.188f);
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__global__ void mandelbrotKernel(
    uchar4* output, int width, int height,
    double centerX, double centerY,
    double scale, int maxIter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double zx = 0.0;
    double zy = 0.0;

    double cx = (x - width / 2.0) * scale + centerX;
    double cy = (y - height / 2.0) * scale + centerY;

    int iter = 0;
    while (zx * zx + zy * zy < 4.0 && iter < maxIter) {
        double tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        iter++;
    }

    float t = iter < maxIter ? (float)iter / maxIter : 0.0f;
    uchar4 color = elegantColor(t);
    output[y * width + x] = color;
}

void launchMandelbrotKernel(
    uchar4* devPtr, int width, int height,
    double centerX, double centerY,
    double scale, int maxIter)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    mandelbrotKernel<<<gridSize, blockSize>>>(
        devPtr, width, height, centerX, centerY, scale, maxIter);
}
