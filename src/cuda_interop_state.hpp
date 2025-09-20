///// Otter: Centralized CUDA/GL interop state; single source of truth for shared globals.
///// Schneefuchs: No heavy headers here; forward-declare CUDA/GL types; helpers only declared.
///// Maus: Deterministic, ASCII-only; header stays light to prevent include-depth blowups.
///// Datei: src/cuda_interop_state.hpp

#pragma once
#include <unordered_map>
#include <cstdint>

// ---- Keep header light: only fwd decls -------------------------------------
struct __GLsync; using GLsync = __GLsync*;
#if !defined(__gl_h_) && !defined(GLEW_H) && !defined(__glew_h__)
  using GLuint = unsigned int;
#endif

// CUDA event (opaque; no cuda_runtime.h in header)
struct CUevent_st; using cudaEvent_t = CUevent_st*;

// PBO resource is defined in the CudaInterop namespace (matches its header)
namespace CudaInterop { class bear_CudaPBOResource; }

namespace CudaInterop {
namespace Detail {

// Shared state (defined once in cuda_interop_state.cpp)
extern CudaInterop::bear_CudaPBOResource* s_pboActive;
extern std::unordered_map<GLuint, CudaInterop::bear_CudaPBOResource*> s_pboMap;

extern bool s_pauseZoom;
extern bool s_deviceOk;

extern cudaEvent_t s_evStart;
extern cudaEvent_t s_evStop;
extern bool        s_evInit;

// Helpers (implemented in cuda_interop_state.cpp)
void ensureDeviceOnce();
void ensureEventsOnce();
void destroyEventsIfAny();
void enforceWriteDiscard(CudaInterop::bear_CudaPBOResource* res);

} // namespace Detail
} // namespace CudaInterop
