///// Otter: Interop header minimal & exact; no hidden globals; ASCII-only.
///// Schneefuchs: Headers/Sources in sync; explicit streams; /WX-safe.
///// Maus: State-driven ownership (render/copy streams passed in).
///// Datei: src/cuda_interop.hpp

#pragma once

#include <cstddef>
#include <vector>               // std::vector<float>
#include <vector_types.h>       // float2 (leichtgewichtig, keine PCH)

// CUDA: nur schlanke Vorwärtsdeklaration (keine schweren cuda_runtime*-Includes)
struct CUstream_st; using cudaStream_t = CUstream_st*;

// Projekt-Typen nur vorwärts deklarieren, um Abhängigkeiten zu reduzieren
namespace Hermelin {
    class CudaDeviceBuffer;
    class GLBuffer;
}
class RendererState;

namespace CudaInterop {

// ---- Sichtbarkeit/Diagnose ---------------------------------------------------
void logCudaDeviceContext(const char* tag);

// ---- CUDA Runtime Precheck / Sanity ------------------------------------------
bool precheckCudaRuntime();

// ---- Pause-Zoom Toggle (Host) ------------------------------------------------
void setPauseZoom(bool pause);
bool getPauseZoom();

// ---- GL-PBO Registrierung ----------------------------------------------------
void registerAllPBOs(const unsigned* pboIds, int count);
void unregisterAllPBOs();
void registerPBO(const Hermelin::GLBuffer& pbo);
void unregisterPBO();

// ---- Kernpfad: Render + Analyse + Host-Spiegel -------------------------------
// Wichtig: Beide Streams EXPLIZIT (Ownership im RendererState)
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
    RendererState& state,
    cudaStream_t renderStream,
    cudaStream_t copyStream
);

} // namespace CudaInterop
