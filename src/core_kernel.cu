// Datei: src/core_kernel.cu
// 🐭 Maus-Kommentar: Mandelbrot-Renderer & adaptive Komplexitätsbewertung — mit dynamischen Kachelgrößen

#include <cstdio>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include "settings.hpp"
#include "core_kernel.h"

// 🐭 Device-Konstante für Variance Threshold (für spätere dynamische Anpassungen)
__device__ float deviceVarianceThreshold = 1e-6f;

// 🐭 Host-Funktion zum Setzen des Variance Thresholds
extern "C" void setDeviceVarianceThreshold(float threshold) {
    cudaMemcpyToSymbol(deviceVarianceThreshold, &threshold, sizeof(float));
}

// 🐭 Debug-Testbild: Farbverlauf statt Mandelbrot
__global__ void testKernel(uchar4* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    img[y * w + x] = make_uchar4((x * 255) / w, (y * 255) / h, 128, 255);
}

// 🐭 Wrapper zum Starten des Testbilds (Zoom-Parameter für spätere Erweiterung vorbereitet)
extern "C" void launch_debugGradient(uchar4* img, int w, int h, float zoom) {
    (void)zoom;  // 🐭 Aktuell unbenutzt — später evtl. Zoom-abhängige Farbmodulation

    dim3 threads(Settings::BASE_TILE_SIZE, Settings::BASE_TILE_SIZE);  // 🐭 Feste Threads für Debug
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    printf("[INFO] DebugGradient Grid (%d, %d)\n", blocks.x, blocks.y);

    testKernel<<<blocks, threads>>>(img, w, h);
    cudaDeviceSynchronize();
}

// 🖌️ Mandelbrot-Color-Mapping (Sanfte Farbverläufe)
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter, float zx, float zy, float zoom) {
    if (iter >= maxIter) return make_uchar4(0, 0, 0, 255);  // Schwarz für Punkte innerhalb der Menge

    float log_zn = logf(zx * zx + zy * zy) * 0.5f;
    float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
    float t = (iter + 1.0f - nu) / maxIter;

    float zoomShift = fmodf(logf(zoom + 2.0f) * 0.07f, 1.0f);

    float r = 0.8f + 0.2f * cosf(6.28318f * (t + zoomShift));
    float g = 0.6f + 0.4f * cosf(6.28318f * (t + zoomShift + 0.3f));
    float b = 0.4f + 0.6f * cosf(6.28318f * (t + zoomShift + 0.6f));

    r = powf(r, 1.5f);
    g = powf(g, 1.5f);
    b = powf(b, 1.5f);

    return make_uchar4(fminf(r * 255.0f, 255.0f), fminf(g * 255.0f, 255.0f), fminf(b * 255.0f, 255.0f), 255);
}

// 🌀 Mandelbrot-Berechnung pro Pixel + Iterationspuffer schreiben
__global__ void mandelbrotHybrid(
    uchar4* img,
    int* iterations,
    int w,
    int h,
    float zoom,
    float2 offset,
    int maxIter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float cx = (x - w * 0.5f) / zoom + offset.x;
    float cy = (y - h * 0.5f) / zoom + offset.y;
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;

    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xt = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = xt;
        ++iter;
    }

    img[y * w + x] = colorMap(iter, maxIter, zx, zy, zoom);
    iterations[y * w + x] = iter;
}

// 🐭 Host-Wrapper für Mandelbrot-Rendering
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,
    int* iterations,
    int w,
    int h,
    float zoom,
    float2 offset,
    int maxIter
) {
    static bool firstLaunch = true;
    dim3 threads(Settings::BASE_TILE_SIZE, Settings::BASE_TILE_SIZE);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    if (firstLaunch) {
        printf("[INFO] Launch mandelbrotHybrid: Grid (%d, %d)\n", blocks.x, blocks.y);
        firstLaunch = false;
    }

    mandelbrotHybrid<<<blocks, threads>>>(img, iterations, w, h, zoom, offset, maxIter);
    cudaDeviceSynchronize();
}

// 🧮 Komplexitätsbewertung mit adaptiver Tile-Size
extern "C" __global__ void computeComplexity(
    const int* iterations,
    int w,
    int h,
    float* complexity,
    int tileSize   // 🐭 Adaptiv!
) {
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tilesX = (w + tileSize - 1) / tileSize;

    int startX = tileX * tileSize;
    int startY = tileY * tileSize;

    int localX = threadIdx.x;
    int localY = threadIdx.y;
    int x = startX + localX;
    int y = startY + localY;

    int localId = localY * blockDim.x + localX;

    extern __shared__ float sharedData[];
    float* sumIter   = sharedData;
    float* sumIterSq = sharedData + blockDim.x * blockDim.y;
    int*   count     = (int*)(sharedData + 2 * blockDim.x * blockDim.y);

    float iterValue = 0.0f;
    int valid = 0;

    if (x < w && y < h) { 
        iterValue = (float)iterations[y * w + x]; 
        valid = 1; 
    }

    sumIter[localId] = iterValue;
    sumIterSq[localId] = iterValue * iterValue;
    count[localId] = valid;

    __syncthreads();

    // 🛠️ Parallel Reduction für Summe und Summe der Quadrate
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sumIter[localId] += sumIter[localId + stride];
            sumIterSq[localId] += sumIterSq[localId + stride];
            count[localId] += count[localId + stride];
        }
        __syncthreads();
    }

    if (localId == 0) {
        int n = count[0];
        if (n > 1) {
            float mean = sumIter[0] / n;
            float meanSq = sumIterSq[0] / n;
            float variance = meanSq - mean * mean;
            variance = variance > 0.0f ? variance : 0.0f; // 🐭 Sicherheit gegen numerischen Fehler
            float stddev = sqrtf(variance);

            complexity[tileY * tilesX + tileX] = stddev;
        } else {
            complexity[tileY * tilesX + tileX] = 0.0f;
        }
    }
}
