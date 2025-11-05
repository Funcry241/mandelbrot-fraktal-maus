///// Otter: Public CUDA interop API - single Capybara render path; no EC baggage; clean declarations.
///// Schneefuchs: Headers & sources in sync; minimal includes; no GL in header; stable signatures.
///// Maus: Expose only what's needed by callers; heavy impl stays in .cu; ASCII-only logs live elsewhere.
///// Datei: src/cuda_interop.hpp

#pragma once

// Keine schweren CUDA-Includes im Header – nur Forward-Decl für cudaStream_t
struct CUstream_st; using cudaStream_t = CUstream_st*;

// Vorwärtsdeklarationen statt schwerer Includes
namespace Hermelin { class CudaDeviceBuffer; }
struct RendererState;   // <— HIER von class -> struct
struct FrameContext;

namespace CudaInterop {

// Runtime environment checks & diagnostics
bool precheckCudaRuntime() noexcept;
void logCudaDeviceContext(const char* tag) noexcept;
void logCudaContext(const char* tag) noexcept; // Alias auf logCudaDeviceContext

// Global pause toggle for zoom logic
void setPauseZoom(bool paused) noexcept;
bool getPauseZoom() noexcept;

// PBO resource lifecycle (IDs sind OpenGL-PBO-IDs)
void registerAllPBOs(const unsigned int* pboIds, int count);
void unregisterAllPBOs() noexcept;

// Render a frame (low-level)
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

// Convenience overload (double Offsets)
void renderCudaFrame(
    RendererState& state,
    const FrameContext& fctx,
    double& newOffsetX,
    double& newOffsetY
);

// GPU-Heatmap
bool buildHeatmapMetrics(RendererState& state,
                         int width, int height, int tilePx,
                         cudaStream_t stream) noexcept;

} // namespace CudaInterop
