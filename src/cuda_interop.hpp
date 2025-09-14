#pragma once
#include <cstdint>
#include <vector>
#include "hermelin_buffer.hpp"
#include "renderer_state.hpp"

using GLuint = unsigned int;

namespace CudaInterop {

void registerAllPBOs(const GLuint* ids, int count);
void unregisterAllPBOs();
void registerPBO(const Hermelin::GLBuffer& pbo);

void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,
    Hermelin::CudaDeviceBuffer& d_contrast,
    int width, int height,
    float zoom, float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,
    float2& newOffset, bool& shouldZoom,
    int tileSize, RendererState& state
);

void setPauseZoom(bool pause);
bool getPauseZoom();

bool precheckCudaRuntime();
bool verifyCudaGetErrorStringSafe();
void unregisterPBO();
void logCudaDeviceContext(const char* tag);

} // namespace CudaInterop
