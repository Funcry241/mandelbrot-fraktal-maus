// mandelbrot_dd.cu – Double-Double Mandelbrot-Kernel mit sanfter Farbgebung, Smooth Coloring, Debugoptionen

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "settings.hpp"
#include "dd_real.cuh"

// Debug-Modus aktivieren (1 = aktiviert)
#define DEBUG_PATTERN 0

__device__ __forceinline__ uchar4 elegantColor(float t) {
    // Sanfter Gold-Blau-Verlauf
    float tSharp = powf(t, 0.5f);  // A) Sanftere Verläufe
    float r = 1.0f - tSharp;
    float g = 0.6f * tSharp;
    float b = 0.4f + 0.5f * tSharp;
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

#if DEBUG_PATTERN
    // D) Testmuster zur GPU-Verifikation
    output[y * width + x] = make_uchar4((x * 5) % 256, (y * 3) % 256, 128, 255);
    return;
#endif

    dd_real zx(0.0), zy(0.0);
    dd_real cx = (dd_real(x) - dd_real(width) / 2.0) * scale + centerX;
    dd_real cy = (dd_real(y) - dd_real(height) / 2.0) * scale + centerY;

    int iter = 0;
    while ((zx * zx + zy * zy).value() < 4.0 && iter < maxIter) {
        dd_real xtemp = zx * zx - zy * zy + cx;
        zy = zx * zy * 2.0 + cy;
        zx = xtemp;
        iter++;
    }

    // B) Smooth Coloring – logarithmisch & stabil
    float zx2 = float(zx.value() * zx.value());
    float zy2 = float(zy.value() * zy.value());
    float mag2 = zx2 + zy2 + 1e-20f;

    float log_zn = logf(mag2) / 2.0f;
    float nu = log2f(log_zn);
    float t = (iter + 1 - nu) / maxIter;
    t = fminf(fmaxf(t, 0.0f), 1.0f);  // Clamp

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

    // E) Optional: Koordinaten fixieren für tiefe Zoomregionen (kommentierbar)
    // dd_real offsetX(-0.743643887037158704752191506114774, 0.00000000000000000000000000000000001);
    // dd_real offsetY( 0.131825904205311970493132056385139, 0.00000000000000000000000000000000001);

    dd_real offsetX(offX_hi, offX_lo);
    dd_real offsetY(offY_hi, offY_lo);
    dd_real scale(zoom);

    mandelbrotKernelDD<<<gridSize, blockSize>>>(devPtr, w, h, offsetX, offsetY, scale, maxIter);
}
