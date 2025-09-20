///// Otter: E/C streaming: add stream+event params; zero host sync; ASCII logs only.
///// Schneefuchs: Backward-compatible defaults; header-only change; no ABI break for callers using defaults.
///// Maus: Event is optional; record at end of E/C chain if non-null; kernels must launch on provided stream.
///// Datei: src/core_kernel.h

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// NOTE [Otter]:
// Public E/C entrypoint now supports async launches.
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

// ============================================================================
// Perturbation (Reference-Orbit) – always-on support
// - Header-only additions; existing callers unaffected.
// - Kernels launched via the new unified entrypoint MUST use the provided stream.
// - No host-side synchronization inside these functions.
// ============================================================================

enum class PertStore : uint8_t { Const = 0, Global = 1 };

// Header passed to the kernel (by value) to describe the active reference orbit.
struct PerturbHeader {
    double2 c_ref;     // reference center in C
    double  pxScale;   // pixel scale (|ΔC| per screen pixel) — from a single canonical source
    int     iterLen;   // total reference-iteration length available in zref buffer
    int     segSize;   // segment size used for building/streaming the orbit
    int     version;   // monotonically increasing version to detect stale uploads
};

// Unified Mandelbrot launch with progressive + perturbation path.
// Requirements:
//  - d_iterations/d_stateZ/d_stateIt must be valid device pointers (may be null if feature disabled).
//  - d_zref must point to device memory containing `zrefCount` elements (double2).
//  - `hdr` is passed by value (copied to kernel params/const regs).
//  - Always launches on `stream`; no host sync inside.
extern "C" void launchMandelbrotUnified(
    uint16_t*       d_iterations,   // [out] per-pixel iteration counts (device)
    float2*         d_stateZ,       // [in/out] progressive Z state (device) or nullptr
    uint16_t*       d_stateIt,      // [in/out] progressive iteration state (device) or nullptr
    int             width,
    int             height,
    double2         center,         // current view center in C
    double2         pixelScale,     // canonical pixel scale (x,y)
    double          zoom,           // current zoom (for telemetry-only if not needed)
    int             maxIterations,
    // --- Perturbation (always-on path expects a valid orbit; callers ensure readiness)
    const double2*  d_zref,         // device pointer to reference-orbit samples
    int             zrefCount,      // number of samples available at d_zref
    PerturbHeader   hdr,            // orbit metadata (by value)
    PertStore       store,          // where d_zref lives (Const/Global) — for telemetry/log-only decisions
    // --- Launch control
    cudaStream_t    stream = nullptr
);
