#pragma once

#include <cuda_runtime.h>
#include <utility>

// CUDA-Kernel-Wrapper (Double-Double-Präzision, GPU-optimiert)
extern "C" void launch_kernel_dd(uchar4* devPtr,
                                  int w, int h,
                                  double zoom,
                                  double offX_hi, double offX_lo,
                                  double offY_hi, double offY_lo,
                                  int maxIter);

// Boundary-Compute auf der GPU (für adaptives Focus-Panning)
// zoom, offX, offY: aktuelle Zoom- und Offset-Werte
// w,h: Bilddimensionen
// sampleStep: Schrittweite für Gradientenabtastung
// maxIter: maximale Iterationsanzahl
std::pair<float,float> computeBoundaryGPU(
    double zoom, double offX, double offY,
    int w, int h,
    int sampleStep, int maxIter
);
