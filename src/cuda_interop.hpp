///// Otter: Public interop header without CUDA vector types; matches .cu signature (no float2 leaks).
///// Schneefuchs: Header/Source in lockstep; minimal includes; forward decls for CUDA stream and project types only.
///// Maus: One-path pipeline (Capybara) exposed via a single render call inside namespace CudaInterop.
///// Datei: src/cuda_interop.hpp

#pragma once

#include <vector> // we use std::vector<float>& in the public API

// Forward declarations to avoid pulling large headers.
class RendererState;
struct FrameContext;

// Hermelin device buffer (to avoid including hermelin_buffer.hpp here)
namespace Hermelin { class CudaDeviceBuffer; }

// Forward-declare CUDA stream type without including cuda_runtime_api.h.
struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace CudaInterop {

// ---- Lifecycle / environment helpers -------------------------------------

// Light-weight CUDA sanity check (device presence / runtime init).
[[nodiscard]] bool precheckCudaRuntime() noexcept;

// Toggle auto-zoom pause flag (global for current process).
void setPauseZoom(bool paused) noexcept;
[[nodiscard]] bool getPauseZoom() noexcept;

// Register/unregister a ring of PBOs for CUDA-GL interop.
void registerAllPBOs(const unsigned int* pboIds, int count);
void unregisterAllPBOs() noexcept;

// Log current CUDA device/runtime context (driver/runtime versions, CC, name).
void logCudaDeviceContext(const char* tag) noexcept;

/**
 * Render one frame into the current PBO using the Capybara path (iterations → colorize → 
 * PBO). This signature is the "low-level" entry and is intentionally explicit about buffers.
 * The RendererState/FrameContext-friendly overload is provided below.
 *
 * Behavior:
 * - Maps the current PBO (based on state.pboIndex), colorizes into it, and unmaps it.
 * - Records timing events internally when performance logging is enabled.
 */
void renderCudaFrame(
    Hermelin::CudaDeviceBuffer& d_iterations,
    Hermelin::CudaDeviceBuffer& d_entropy,    // legacy EC buffer (unused in render-only path)
    Hermelin::CudaDeviceBuffer& d_contrast,   // legacy EC buffer (unused in render-only path)
    int   width,
    int   height,
    float zoom,
    float offsetX,
    float offsetY,
    int   maxIterations,
    std::vector<float>& h_entropy,            // legacy EC host buffer (unused)
    std::vector<float>& h_contrast,           // legacy EC host buffer (unused)
    float& newOffsetX,
    float& newOffsetY,
    bool&  shouldZoom,
    int    tileSize,
    RendererState& state,
    cudaStream_t renderStream,
    cudaStream_t copyStream
);

// Convenience overload used by the frame pipeline.
void renderCudaFrame(RendererState& state, const FrameContext& fctx, float& newOffsetX, float& newOffsetY);

} // namespace CudaInterop
