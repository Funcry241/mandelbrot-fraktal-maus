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
extern "C" void computeCudaEntropyContrast(
    const uint16_t* d_iterations,  // per-pixel iteration counts (device)
    float*          d_entropyOut,  // per-tile entropy (device)
    float*          d_contrastOut, // per-tile contrast (device)
    int             width,
    int             height,
    int             tileSize,
    int             maxIterations,
    cudaStream_t    stream      = nullptr,  // default stream (legacy)
    cudaEvent_t     ecDoneEvent = nullptr   // no event by default
);
