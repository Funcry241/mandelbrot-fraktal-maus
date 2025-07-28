// 🐭 Maus-Kommentar: Alpha 68½ - Früherkennung von CUDA-Geräten + sicherer API-Test für Fehlerauflösung.
// Otter: fail fast. Schneefuchs: keine Überraschungen. Maus: aber elegant.

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <GLFW/glfw3.h>
#include <vector_types.h>
#include "core_kernel.h" // extern "C" computeCudaEntropyContrast

class RendererState;

namespace CudaInterop {

void registerPBO(unsigned int pbo);
void unregisterPBO();

// Haupt-Renderfunktion für CUDA-Mandelbrot mit Analyse & Heatmap
void renderCudaFrame(
    int* d_iterations,
    float* d_entropy,
    float* d_contrast,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize,
    RendererState& state
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// Alpha 68: Früher Check für CUDA-Verfügbarkeit - nur numerische Codes
bool precheckCudaRuntime(); // 🐭 Schneefuchs: keine Devices = kein Rendererstart

// Alpha 68½: Früher Test, ob cudaGetErrorString gefahrlos aufrufbar ist
bool verifyCudaGetErrorStringSafe(); // 🐭 Otter: Nur im Precheck erlaubt!

// Capybara: Inline-Wrapper für extern "C" Kernel (core_kernel.cu)
inline void computeCudaEntropyContrast(
    const int* d_iterations,
    float* d_entropyOut,
    float* d_contrastOut,
    int width,
    int height,
    int tileSize,
    int maxIter
) {
    ::computeCudaEntropyContrast(d_iterations, d_entropyOut, d_contrastOut, width, height, tileSize, maxIter);
}

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
