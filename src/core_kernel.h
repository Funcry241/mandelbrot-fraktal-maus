///// Otter: E/C streaming: add stream+event params; zero host sync; ASCII logs only.
///// Schneefuchs: Backward-compatible defaults; header-only change; no ABI break for callers using defaults.
///// Maus: Event is optional; record at end of E/C chain if non-null; kernels must launch on provided stream.
/// /// Datei: src/core_kernel.h

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// NOTE [Otter]: Public E/C entrypoint now supports async launches.
// - No host-side synchronization inside this function.
// - Both entropy and contrast kernels must be launched on `stream`.
// - If `ecDoneEvent` is provided (non-null), it is recorded on `stream` after the last E/C kernel.
// - Callers that omit the new arguments keep previous behavior (default stream, no event).
//
// Parameters:
//   d_iterations   : [in]  per-pixel iteration counts (device memory)
//   d_entropyOut   : [out] per-tile/pixel entropy output (device memory; exact layout as before)
//   d_contrastOut  : [out] per-tile/pixel contrast output (device memory; exact layout as before)
//   width,height   : [in]  image dimensions in pixels
//   tileSize       : [in]  tile size used by the analysis (unchanged semantics)
//   maxIterations  : [in]  max iteration used during render (passed through for contrast heuristics)
//   stream         : [in]  CUDA stream for kernel launches (default=0 → legacy default stream)
//   ecDoneEvent    : [in]  optional; if non-null, will be cudaEventRecord(...) after contrast kernel
extern "C" void computeCudaEntropyContrast(
    const uint16_t* d_iterations,
    float*          d_entropyOut,
    float*          d_contrastOut,
    int             width,
    int             height,
    int             tileSize,
    int             maxIterations,
    cudaStream_t    stream       /*= nullptr*/,
    cudaEvent_t     ecDoneEvent  /*= nullptr*/
);
