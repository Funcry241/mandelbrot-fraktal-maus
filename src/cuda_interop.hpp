// Datei: src/cuda_interop.hpp
// Zeilen: 61
// 🐭 Maus-Kommentar: Schnittstelle CUDA/OpenGL – Kolibri, Panda & Capybara 2 sauber integriert, float2-only.
// Wrapper für Host/Device-Puffer, Heatmap & Auto-Zoom. Keine Redundanz – alles klar Otter!

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
    int supersampling,
    RendererState& state,
    int* d_tileSupersampling,
    std::vector<int>& h_tileSupersampling
);

void setPauseZoom(bool pause);
bool getPauseZoom();

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
