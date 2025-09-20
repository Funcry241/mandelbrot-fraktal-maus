///// Otter: E/C streaming: add stream+event params; zero host sync; ASCII logs only.
///// Schneefuchs: Backward-compatible defaults; header-only change; no ABI break for callers using defaults.
///// Maus: Event is optional; record at end of E/C chain if non-null; kernels must launch on provided stream.
///// Datei: src/core_kernel.h

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "settings.hpp" // for Settings::zrefMaxLen

// NOTE [Otter]:
// Public E/C entrypoint now supports async launches.
extern "C" void computeCudaEntropyContrast(
    const uint16_t* d_iterations,
    float*          d_entropyOut,
    float*          d_contrastOut,
    int             width,
    int             height,
    int             tileSize,
    int             maxIterations,
    cudaStream_t    stream      = nullptr,
    cudaEvent_t     ecDoneEvent = nullptr
);

// ============================================================================
// Perturbation (Reference-Orbit) – always-on support
// ============================================================================

enum class PertStore : uint8_t { Const = 0, Global = 1 };

// MSVC C4324 (padding due to alignment) – disable locally under /WX.
#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable:4324)
#endif

// Contract: if p.active == 0 → classical path (ignore others)
struct PerturbParams final {
    int        active;
    int        len;
    int        segSize;
    PertStore  store;
    double2    c_ref;
    double     deltaGuard;
    int        version;
};

struct PerturbHeader {
    double2 c_ref;
    double  pxScale;
    int     iterLen;
    int     segSize;
    int     version;
};

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

// CONST reference-orbit buffer (device). Must match the definition in core_kernel.cu.
extern __constant__ double2 zrefConst[Settings::zrefMaxLen];

// PERT telemetry symbol (device): max |delta| per frame, updated by kernel.
extern __device__ float d_deltaMax;

// Optional unified Mandelbrot launch (declaration only).
extern "C" void launchMandelbrotUnified(
    uint16_t*       d_iterations,
    float2*         d_stateZ,
    uint16_t*       d_stateIt,
    int             width,
    int             height,
    double2         center,
    double2         pixelScale,
    double          zoom,
    int             maxIterations,
    const double2*  d_zref,
    int             zrefCount,
    PerturbHeader   hdr,
    PertStore       store,
    cudaStream_t    stream = nullptr
);
