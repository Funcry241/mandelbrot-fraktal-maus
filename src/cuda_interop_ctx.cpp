///// Otter: Context utilities: pause zoom flag, runtime precheck, device context log.
///// Schneefuchs: Clean split from hot path; robust attribute queries; safe fallbacks.
///// Maus: ASCII-only; numeric error codes; zero noise when logs are disabled.
///// Datei: src/cuda_interop_ctx.cpp

#include "pch.hpp"
#include "cuda_interop.hpp"
#include "cuda_interop_state.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"
#include <cstdlib>   // _TRUNCATE
#include <string.h>  // strncpy_s
#include <cuda_runtime.h>
#include <cstring>

namespace CudaInterop {
using namespace Detail;

void setPauseZoom(bool pause) { s_pauseZoom = pause; }
bool getPauseZoom()           { return s_pauseZoom; }

bool precheckCudaRuntime() {
    int deviceCount = 0;
    cudaError_t e1 = cudaFree(0);
    cudaError_t e2 = cudaGetDeviceCount(&deviceCount);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CUDA] precheck err1=%d err2=%d count=%d", (int)e1, (int)e2, deviceCount);
    }
    return e1 == cudaSuccess && e2 == cudaSuccess && deviceCount > 0;
}

static inline int getAttrSafe(cudaDeviceAttr a, int dev) {
    int v = 0; (void)cudaDeviceGetAttribute(&v, a, dev); return v;
}

void logCudaDeviceContext(const char* tag) {
    if constexpr (!(Settings::debugLogging || Settings::performanceLogging)) { (void)tag; return; }
    int dev = -1; cudaError_t e0 = cudaGetDevice(&dev);
    int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    char name[256] = {0};

    if (e0 == cudaSuccess && dev >= 0) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
            ::strncpy_s(name, sizeof(name), p.name, _TRUNCATE);  // safe, no C4996
        }
        const int ccM = getAttrSafe(cudaDevAttrComputeCapabilityMajor, dev);
        const int ccN = getAttrSafe(cudaDevAttrComputeCapabilityMinor, dev);
        const int sms = getAttrSafe(cudaDevAttrMultiProcessorCount, dev);
        const int warp = getAttrSafe(cudaDevAttrWarpSize, dev);
        size_t mf = 0, mt = 0; cudaMemGetInfo(&mf, &mt);

        LUCHS_LOG_HOST("[CUDA] ctx tag=%s rt=%d drv=%d dev=%d name=\"%s\" cc=%d.%d sms=%d warp=%d memMB free=%zu total=%zu",
            (tag ? tag : "(null)"), rt, drv, dev, name, ccM, ccN, sms, warp, (mf >> 20), (mt >> 20));
    } else {
        LUCHS_LOG_HOST("[CUDA] ctx tag=%s deviceQuery failed e0=%d dev=%d",
            (tag ? tag : "(null)"), (int)e0, dev);
    }
}

} // namespace CudaInterop
