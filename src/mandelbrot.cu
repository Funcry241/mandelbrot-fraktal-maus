// src/mandelbrot.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Wandelt einen Wert h∈[0,1) in RGB um (HSV-Farbkreis, S=1,V=1)
__device__ uchar4 hsv2rgb(float h) {
    float r, g, b;
    float i = floorf(h * 6.0f);
    float f = h * 6.0f - i;
    float q = 1.0f - f;

    switch (int(i) % 6) {
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        case 5: r = 1; g = 0; b = q; break;
    }
    return make_uchar4(
        static_cast<unsigned char>(r * 255.0f),
        static_cast<unsigned char>(g * 255.0f),
        static_cast<unsigned char>(b * 255.0f),
        255u
    );
}

extern "C"
__global__ void mandelbrotKernel(uchar4* output,
                                 int width, int height,
                                 double zoom,
                                 double offX, double offY,
                                 int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Berechne komplexe Koordinaten
    double cx = ( (double)x / width  - 0.5) * 3.5 / zoom + offX;
    double cy = ( (double)y / height - 0.5) * 2.0 / zoom + offY;

    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx*zx + zy*zy <= 4.0 && iter < maxIter) {
        double xt = zx*zx - zy*zy + cx;
        zy = 2.0*zx*zy + cy;
        zx = xt;
        ++iter;
    }

    // Normierte Iterationszahl für Farbgebung
    float t = iter < maxIter
            ? (float)iter / maxIter
            : 0.0f; // Innen schwarz

    output[y * width + x] = (iter < maxIter)
        ? hsv2rgb(t)       // Farbverlauf am Rand
        : make_uchar4(0,0,0,255); // Innen komplett schwarz
}

extern "C"
void launch_kernel_dd(uchar4* devPtr,
                      int w, int h,
                      double zoom,
                      double offX, double offY,
                      int maxIter)
{
    dim3 block(16,16);
    dim3 grid((w + block.x - 1)/block.x,
              (h + block.y - 1)/block.y);
    mandelbrotKernel<<<grid, block>>>(devPtr, w, h, zoom, offX, offY, maxIter);
    cudaDeviceSynchronize();
}
