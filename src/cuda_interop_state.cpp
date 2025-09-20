///// Otter: Defines shared interop state + helper implementations used by all TUs.
///// Schneefuchs: Single definition to satisfy ODR; minimal includes; no GL headers pulled in.
///// Maus: Events are created/destroyed deterministically; numeric CUDA rc only.
///// Datei: src/cuda_interop_state.cpp

#include "pch.hpp"
#include "cuda_interop_state.hpp"

#include "luchs_log_host.hpp"
#include "settings.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace CudaInterop {
namespace Detail {

// ---- Shared state (single definitions) --------------------------------------
bear_CudaPBOResource* s_pboActive = nullptr;
std::unordered_map<GLuint, bear_CudaPBOResource*> s_pboMap{};

bool s_pauseZoom = false;
bool s_deviceOk  = false;

cudaEvent_t s_evStart = nullptr;
cudaEvent_t s_evStop  = nullptr;
bool        s_evInit  = false;

// ---- Helpers ----------------------------------------------------------------
void ensureDeviceOnce() {
    if (!s_deviceOk) {
        CUDA_CHECK(cudaSetDevice(0));
        s_deviceOk = true;
    }
}

void ensureEventsOnce() {
    if (s_evInit) return;
    CUDA_CHECK(cudaEventCreate(&s_evStart));
    CUDA_CHECK(cudaEventCreate(&s_evStop));
    s_evInit = (s_evStart && s_evStop);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CUDA][ZK] events %s", s_evInit ? "created" : "FAILED");
    }
}

void destroyEventsIfAny() {
    if (!s_evInit) return;
    cudaEventDestroy(s_evStart); s_evStart = nullptr;
    cudaEventDestroy(s_evStop);  s_evStop  = nullptr;
    s_evInit = false;
}

void enforceWriteDiscard(bear_CudaPBOResource* res) {
    if (!res) return;
    if (auto* gr = res->get()) {
        (void)cudaGraphicsResourceSetMapFlags(gr, cudaGraphicsMapFlagsWriteDiscard);
    }
}

} // namespace Detail
} // namespace CudaInterop
