///// Otter: Public CUDA interop API — single Capybara render path; no EC baggage; clean declarations.
///// Schneefuchs: Headers & sources in sync; minimal includes; no GL in header; stable signatures.
///// Maus: Expose only what's needed by callers; heavy impl stays in .cu; ASCII-only logs live elsewhere.
///// Datei: src/cuda_interop.hpp

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "hermelin_buffer.hpp"   // Hermelin::CudaDeviceBuffer
#include "renderer_state.hpp"    // RendererState
#include "frame_context.hpp"     // FrameContext

namespace CudaInterop {

// Runtime environment checks & diagnostics
bool precheckCudaRuntime() noexcept;
void logCudaDeviceContext(const char* tag) noexcept;
void logCudaContext(const char* tag) noexcept; // alias to logCudaDeviceContext

// Global pause toggle for zoom logic
void setPauseZoom(bool paused) noexcept;
bool getPauseZoom() noexcept;

// PBO resource lifecycle
void registerAllPBOs(const unsigned int* pboIds, int count);
void unregisterAllPBOs() noexcept;

// Render a frame (low-level variant for internal callers)
void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    int   width,
    int   height,
    float zoom,
    float offsetX,
    float offsetY,
    int   maxIterations,
    float& newOffsetX,
    float& newOffsetY,
    bool&  shouldZoom,
    RendererState& state,
    cudaStream_t renderStream
);

// Convenience overload used by the main renderer loop
void renderCudaFrame(
    RendererState& state,
    const FrameContext& fctx,
    float& newOffsetX,
    float& newOffsetY
);

} // namespace CudaInterop
