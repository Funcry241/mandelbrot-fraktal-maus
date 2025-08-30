// Datei: src/cuda_interop.hpp
// 🐻 Bär: RAII-Integration für CUDA-PBO im Projekt „Bär“.

#pragma once
#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <vector_types.h>         // float2
#include "core_kernel.h"          // computeCudaEntropyContrast(...)
#include "hermelin_buffer.hpp"    // Hermelin::CudaDeviceBuffer, Hermelin::GLBuffer

class RendererState;

namespace CudaInterop {

// 🐻 Bär: Registriert den PBO als CUDA-Resource (RAII-Wrapper intern)
void registerPBO(const Hermelin::GLBuffer& pbo);

// (unregister erfolgt automatisch über RAII in bear_CudaPBOResource, hier nur Cleanup-Funktion:)
void unregisterPBO();

// 🐻 Bär: Haupt-Renderfunktion …
void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
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
[[nodiscard]] bool getPauseZoom();
[[nodiscard]] bool precheckCudaRuntime();
[[nodiscard]] bool verifyCudaGetErrorStringSafe();
void logCudaDeviceContext(const char* context);

// Bequemer Inline-Forwarder für die E/C-Pipeline (keine neue ABI-Oberfläche).
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
