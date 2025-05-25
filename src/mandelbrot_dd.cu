// mandelbrot_dd.cu – Double-Double Mandelbrot-Kernel mit sanfter Farbgebung

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "settings.hpp"
#include "dd_real.cuh" // Annahme: dd_real für Double-Double ist vorhanden

__device__ uchar4 elegantColor(float t) {
    float r = 0.5f + 0.5f * sinf(6.2831f * t);
    float g = 0.5f + 0.5f * sinf(6.2831f * t + 2.094f);
    float b = 0.5f + 0.5f * sinf(6.2831f * t + 4.188f);
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__global__ void mandelbrotKernelDD(
    uchar4* output, int width, int height,
    dd_real centerX, dd_real centerY,
    dd_real scale, int maxIter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    dd_real zx(0.0), zy(0.0);
    dd_real cx = (dd_real(x) - dd_real(width) / 2.0) * scale + centerX;
    dd_real cy = (dd_real(y) - dd_real(height) / 2.0) * scale + centerY;

    int iter = 0;
    while ((zx * zx + zy * zy).hi() < 4.0 && iter < maxIter) {
        dd_real xtemp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xtemp;
        iter++;
    }

    float t = (iter < maxIter) ? float(iter) / maxIter : 0.0f;
    output[y * width + x] = elegantColor(t);
}

extern "C" void launch_kernel_dd(uchar4* devPtr, int w, int h,
                                  double zoom,
                                  double offX_hi, double offX_lo,
                                  double offY_hi, double offY_lo,
                                  int maxIter) {
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                  (h + blockSize.y - 1) / blockSize.y);

    dd_real offsetX(offX_hi, offX_lo);
    dd_real offsetY(offY_hi, offY_lo);
    dd_real scale(zoom);

    mandelbrotKernelDD<<<gridSize, blockSize>>>(devPtr, w, h, offsetX, offsetY, scale, maxIter);
}
