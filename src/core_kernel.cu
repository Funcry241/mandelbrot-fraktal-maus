// Datei: src/core_kernel.cu
// Maus-Kommentar: Implementierung des Hybrid‐Mandelbrot‐Kernels mit adaptiven Kacheln,
//                Thrust‐basierter Komplexitätsberechnung und gründlichem CUDA‐Error‐Handling.

#include "core_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// ----------------------------------------------------------------------
// Dynamischer Schwellenwert für Verschachtelung: Durchschnittliche Iterationen pro Pixel
#ifndef DYNAMIC_THRESHOLD
#define DYNAMIC_THRESHOLD 100.0f
#endif

// ----------------------------------------------------------------------
// Farb‐Mapping: Einfache Palette, damit man Farbunterschiede direkt erkennt
__device__ __forceinline__ uchar4 colorMap(int iter, int maxIter) {
    if (iter == maxIter) {
        // Punkte, die niemals ausbrechen → Rot
        return make_uchar4(255, 0, 0, 255);
    }
    // Linearer Gradient von Blau (niedrige Iteration) nach Grün (hohe Iteration)
    unsigned char c = static_cast<unsigned char>((iter * 255) / maxIter);
    return make_uchar4(c/2, 255 - c, c, 255);
}

// ----------------------------------------------------------------------
// Nested Kernel: Verfeinerung einer Kachel mit doppelter Iterationszahl
__global__ void refineTile(
    uchar4* img, int width, int height,
    float zoom, float2 offset,
    int startX, int startY,
    int tileW, int tileH,
    int maxIter
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tileW || ty >= tileH) return;

    int x = startX + tx;
    int y = startY + ty;
    if (x >= width || y >= height) return;

    // Complex‐Koordinaten berechnen
    float cx = (x - width * 0.5f) / zoom + offset.x;
    float cy = (y - height * 0.5f) / zoom + offset.y;

    // Mandelbrot‐Iteration
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx*zx + zy*zy < 4.0f && iter < maxIter) {
        float xt = zx*zx - zy*zy + cx;
        zy = 2.0f*zx*zy + cy;
        zx = xt;
        ++iter;
    }
    img[y * width + x] = colorMap(iter, maxIter);
}

// ----------------------------------------------------------------------
// Haupt‐Kernel: Tile‐paralleles Mandelbrot mit adaptiver Rekursion
__global__ void mandelbrotHybrid(
    uchar4* img,
    int width, int height,
    float zoom, float2 offset,
    int maxIter
) {
    // Berechne Kachelkoordinaten
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;

    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX = min(startX + TILE_W, width);
    int endY = min(startY + TILE_H, height);

    // Lokale Reduktion: Summe der Iterationen und Pixelanzahl
    float sumIter = 0.0f;
    int cntPix = 0;

    // Thread‐strided Loop, um in der Kachel alle Pixel zu berechnen
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            // Complex‐Koordinaten
            float cx = (x - width * 0.5f) / zoom + offset.x;
            float cy = (y - height * 0.5f) / zoom + offset.y;
            float zx = 0.0f, zy = 0.0f;
            int iter = 0;
            while (zx*zx + zy*zy < 4.0f && iter < maxIter) {
                float xt = zx*zx - zy*zy + cx;
                zy = 2.0f*zx*zy + cy;
                zx = xt;
                ++iter;
            }
            sumIter += iter;
            ++cntPix;
            img[y * width + x] = colorMap(iter, maxIter);
        }
    }

    // Thread (0,0) pro Block entscheidet über Nested‐Launch
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float avgIter = sumIter / cntPix;
        if (avgIter > DYNAMIC_THRESHOLD) {
            int tileW = endX - startX;
            int tileH = endY - startY;
            dim3 bs(min(tileW, TILE_W), min(tileH, TILE_H));
            dim3 gs((tileW + bs.x - 1) / bs.x,
                    (tileH + bs.y - 1) / bs.y);
            refineTile<<<gs, bs>>>(
                img, width, height,
                zoom, offset,
                startX, startY,
                tileW, tileH,
                maxIter * 2
            );
            // Nach jedem Nested‐Kernel auf Error prüfen
            cudaError_t errNested = cudaGetLastError();
            if (errNested != cudaSuccess) {
                printf("refineTile‐Fehler: %s\n", cudaGetErrorString(errNested));
            }
        }
    }
}

// ----------------------------------------------------------------------
// Thrust‐Functor: Liefert 1.0f, wenn Pixel nicht schwarz (Alpha !== 0),
// sonst 0.0f. Wird verwendet in computeComplexity für transform_reduce.
struct IsNonBlack {
    __host__ __device__
    float operator()(const uchar4& pixel) const {
        // Pixel gilt als „nicht schwarz“, wenn mindestens einer der RGB‐Kanäle >0
        return (pixel.x != 0 || pixel.y != 0 || pixel.z != 0) ? 1.0f : 0.0f;
    }
};

// ----------------------------------------------------------------------
// Complexity‐Kernel‐Wrapper: Hier wird Thrust eingesetzt, um pro Tile die Anzahl
// nicht‐schwarzer Pixel zu ermitteln und in das Device‐Array „complexity“ zu schreiben.
// Da Thrust in device‐side‐Kontext oft kompliziert aufzurufen ist, starten wir pro Tile
// einen eigenen CUDA‐Stream und schieben eine Thrust‐Operation auf diesen Stream.
// Achtung: Das ist für größere Tile‐Anzahlen teuer; sollte man später ggf. nur einmalig oder
// in Streams pro Zeile packen. Für Demo‐Zwecke aber ausreichend.
extern "C" __global__ void computeComplexity(
    const uchar4* img,
    int width, int height,
    float* complexity   // Länge: tilesX * tilesY
) {
    // Bestimme Kachel
    int tileX = blockIdx.x * blockDim.x + threadIdx.x;
    int tileY = blockIdx.y * blockDim.y + threadIdx.y;
    int tilesX = (width + TILE_W - 1) / TILE_W;
    int tilesY = (height + TILE_H - 1) / TILE_H;
    int idx = tileY * tilesX + tileX;
    if (tileX >= tilesX || tileY >= tilesY) return;

    // Begrenze Koordinaten für diese Kachel
    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX = min(startX + TILE_W, width);
    int endY = min(startY + TILE_H, height);

    // Gesamtanzahl Pixel in der Kachel
    int tileW = endX - startX;
    int tileH = endY - startY;
    int tileSize = tileW * tileH;
    if (tileSize <= 0) {
        complexity[idx] = 0.0f;
        return;
    }

    // Erzeuge Device‐Zeiger auf den ersten Pixel der Kachel
    const uchar4* basePtr = img + (startY * width + startX);

    // Wir möchten Zeilenweise übergeben, da die Bilddaten „row‐major“ liegen.
    // Thrust kann nicht direkt einen zweidimensionalen Bereich auf einmal verarbeiten,
    // deshalb iterieren wir Zeile für Zeile im Host‐Code – aber da wir in einem __global__‐Kernel
    // sitzen, können wir nur device‐side Thrust aufrufen. Um dennoch eine Zeile nach der anderen zu
    // zählen, brechen wir das Problem pro Tile nicht weiter auf – wir nutzen stattdessen
    // thrust::transform_reduce mit Stride = width, wobei wir nur tileW Elemente/Zeile zählen.

    // Thrust‐Device‐Pointer auf das erste Pixel im Tile
    thrust::device_ptr<const uchar4> devPtr(basePtr);

    // Funktor erzeugen
    IsNonBlack functor;

    // Summiere über jede Zeile separat und addiere zur tileSum
    float tileSum = 0.0f;
    for (int row = 0; row < tileH; ++row) {
        // Zeiger auf den Zeilenanfang in der globalen Bildmatrix
        thrust::device_ptr<const uchar4> rowPtr = devPtr + static_cast<long long>(row) * width;
        // transform_reduce über genau tileW Pixel
        float rowCount = thrust::transform_reduce(
            thrust::device,         // Device‐Kontext
            rowPtr,                 // Start der Zeile
            rowPtr + tileW,         // Ende der Zeile (Exclusive)
            functor,                // transform‐Functor (IsNonBlack)
            0.0f,                   // Initialwert
            thrust::plus<float>()   // Summierer
        );
        tileSum += rowCount;
    }

    complexity[idx] = tileSum;
}

// ----------------------------------------------------------------------
// Launch‐Wrapper für den Hybrid‐Kernel und Fehlerprüfung
extern "C" void launch_mandelbrotHybrid(
    uchar4* img,
    int width, int height,
    float zoom, float2 offset,
    int maxIter
) {
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim(
        (width  + TILE_W - 1) / TILE_W,
        (height + TILE_H - 1) / TILE_H
    );

    // Starte den Haupt‐Kernel
    mandelbrotHybrid<<<gridDim, blockDim>>>(img, width, height, zoom, offset, maxIter);
    cudaError_t errLaunch = cudaGetLastError();
    if (errLaunch != cudaSuccess) {
        printf("mandelbrotHybrid‐Launch‐Error: %s\n", cudaGetErrorString(errLaunch));
    }
}
