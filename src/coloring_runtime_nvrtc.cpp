///// Otter: Nacktmull — NVRTC grayscale JIT path with safe fallback and ASCII telemetry
///// Schneefuchs: Minimal diff; compile-time guard; deterministic logs; no behavior change when OFF
///// Maus: Build once → reuse; measure (compile/load/launch); fallback on any error
///// Datei: src/coloring_runtime_nvrtc.cpp

#include "pch.hpp"
#include "coloring_runtime_nvrtc.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

#if OTTER_USE_NVRTC
  #include <cuda.h>            // Driver API for module/function/launch
  #include <cuda_runtime.h>    // For device props / stream interop
  #include <nvrtc.h>           // Runtime compilation
  #include <chrono>
  #include <string>
  #include <vector>
  #include <cstdio>
  #include <cstring>
#endif

namespace ColoringNVRTC {

bool launch_if_active(const uint16_t* d_it,
                      uchar4*         d_out,
                      int             w,
                      int             h,
                      int             maxIter,
                      cudaStream_t    stream)
{
    (void)d_it; (void)d_out; (void)w; (void)h; (void)maxIter; (void)stream;

#if !OTTER_USE_NVRTC
    // NVRTC entirely disabled at build time → always fall back.
    return false;
#else
    if constexpr (!(Settings::Luchs::enabled && Settings::Luchs::nvrtc)) {
        // Runtime switch is OFF → fall back.
        return false;
    }

    // --- Helpers (local lambdas) ------------------------------------------------
    auto to_ms = [](auto dt) -> double {
        using namespace std::chrono;
        return duration_cast<duration<double, std::milli>>(dt).count();
    };

    auto cu_err = [](CUresult r) -> const char* {
        const char* name = nullptr;
        if (cuGetErrorName(r, &name) != CUDA_SUCCESS || !name) return "CUDA_ERROR_UNKNOWN";
        return name;
    };

    auto cu_str = [](CUresult r) -> const char* {
        const char* s = nullptr;
        if (cuGetErrorString(r, &s) != CUDA_SUCCESS || !s) return "unknown";
        return s;
    };

    auto compute_arch_opt = []() -> std::string {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return std::string("--gpu-architecture=compute_70");
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return std::string("--gpu-architecture=compute_70");
        char buf[64];
        std::snprintf(buf, sizeof(buf), "--gpu-architecture=compute_%d%d", prop.major, prop.minor);
        return std::string(buf);
    };

    // --- One-time JIT (static cache) -------------------------------------------
    static bool       s_ready   = false;
    static CUmodule   s_module  = nullptr;
    static CUfunction s_kernel  = nullptr;
    static std::string s_arch;

    if (!s_ready) {
        const char* kSrc = R"(
extern "C" {

typedef struct { unsigned char x,y,z,w; } uchar4;
__device__ __forceinline__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w){
    uchar4 r = {x,y,z,w}; return r;
}

__global__ void color_kernel(const unsigned short* __restrict__ it,
                             uchar4* __restrict__ out,
                             int width, int height, int maxIter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    unsigned short v = it[idx];
    float g = (maxIter > 0) ? (float)v / (float)maxIter : 0.0f;
    if (g < 0.f) g = 0.f;
    if (g > 1.f) g = 1.f;
    unsigned char c = (unsigned char)(g * 255.0f + 0.5f);

    out[idx] = make_uchar4(c, c, c, 255);
}

} // extern "C"
)";

        nvrtcProgram prog = nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();
        nvrtcResult nrc = nvrtcCreateProgram(&prog, kSrc, "color_kernel.cu", 0, nullptr, nullptr);
        if (nrc != NVRTC_SUCCESS) {
            LUCHS_LOG_HOST("[REPL/COLOR] NVRTC create failed code=%d msg=%s -> fallback", (int)nrc, nvrtcGetErrorString(nrc));
            return false;
        }

        s_arch = compute_arch_opt();
        std::vector<const char*> opts;
        opts.push_back(s_arch.c_str());
        opts.push_back("--std=c++11");

        nrc = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
        auto t1 = std::chrono::high_resolution_clock::now();

        // Capture compile log (even on success) to aid diagnostics
        size_t logSize = 0;
        (void)nvrtcGetProgramLogSize(prog, &logSize);
        std::string log; log.resize(logSize ? logSize - 1 : 0);
        if (logSize > 1) (void)nvrtcGetProgramLog(prog, log.data());

        if (nrc != NVRTC_SUCCESS) {
            LUCHS_LOG_HOST("[REPL/COLOR] NVRTC compile failed code=%d msg=%s arch=%s ms=%.3f log-bytes=%zu -> fallback",
                           (int)nrc, nvrtcGetErrorString(nrc), s_arch.c_str(), to_ms(t1 - t0), logSize);
            if (!log.empty()) {
                // Trim to a single line (ASCII) to keep logs compact
                for (char& ch : log) if (ch == '\n' || ch == '\r') ch = ' ';
                LUCHS_LOG_HOST("[REPL/COLOR] NVRTC log: %s", log.c_str());
            }
            (void)nvrtcDestroyProgram(prog);
            return false;
        }

        size_t ptxSize = 0;
        nrc = nvrtcGetPTXSize(prog, &ptxSize);
        if (nrc != NVRTC_SUCCESS || ptxSize == 0) {
            LUCHS_LOG_HOST("[REPL/COLOR] NVRTC get PTX size failed code=%d msg=%s -> fallback", (int)nrc, nvrtcGetErrorString(nrc));
            (void)nvrtcDestroyProgram(prog);
            return false;
        }

        std::vector<char> ptx; ptx.resize(ptxSize);
        nrc = nvrtcGetPTX(prog, ptx.data());
        (void)nvrtcDestroyProgram(prog);
        if (nrc != NVRTC_SUCCESS) {
            LUCHS_LOG_HOST("[REPL/COLOR] NVRTC get PTX failed code=%d msg=%s -> fallback", (int)nrc, nvrtcGetErrorString(nrc));
            return false;
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        // Ensure a driver context exists
        CUresult cr = cuInit(0);
        if (cr != CUDA_SUCCESS) {
            LUCHS_LOG_HOST("[REPL/COLOR] cuInit failed code=%d name=%s str=%s -> fallback", (int)cr, cu_err(cr), cu_str(cr));
            return false;
        }
        CUcontext ctx = nullptr;
        cr = cuCtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) {
            LUCHS_LOG_HOST("[REPL/COLOR] cuCtxGetCurrent failed code=%d name=%s str=%s -> fallback", (int)cr, cu_err(cr), cu_str(cr));
            return false;
        }

        cr = cuModuleLoadData(&s_module, ptx.data());
        if (cr != CUDA_SUCCESS) {
            LUCHS_LOG_HOST("[REPL/COLOR] cuModuleLoadData failed code=%d name=%s str=%s -> fallback", (int)cr, cu_err(cr), cu_str(cr));
            return false;
        }

        cr = cuModuleGetFunction(&s_kernel, s_module, "color_kernel");
        if (cr != CUDA_SUCCESS) {
            LUCHS_LOG_HOST("[REPL/COLOR] cuModuleGetFunction failed code=%d name=%s str=%s -> fallback", (int)cr, cu_err(cr), cu_str(cr));
            (void)cuModuleUnload(s_module); s_module = nullptr;
            return false;
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        LUCHS_LOG_HOST("[REPL/COLOR] NVRTC ready arch=%s compile-ms=%.3f ptx-bytes=%zu load-ms=%.3f",
                       s_arch.c_str(), to_ms(t1 - t0), ptxSize, to_ms(t3 - t2));
        s_ready = true;
    }

    // --- Launch -----------------------------------------------------------------
    if (!s_ready || s_kernel == nullptr) {
        LUCHS_LOG_HOST("[REPL/COLOR] NVRTC not ready -> fallback");
        return false;
    }

    const int bx = 16, by = 16;
    const int gx = (w + bx - 1) / bx;
    const int gy = (h + by - 1) / by;

    CUdeviceptr arg_it  = reinterpret_cast<CUdeviceptr>(d_it);
    CUdeviceptr arg_out = reinterpret_cast<CUdeviceptr>(d_out);
    int arg_w = w, arg_h = h, arg_max = maxIter;

    void* params[] = { &arg_it, &arg_out, &arg_w, &arg_h, &arg_max };

    CUstream cuStream = reinterpret_cast<CUstream>(stream);

    auto tL0 = std::chrono::high_resolution_clock::now();
    CUresult cr = cuLaunchKernel(
        s_kernel,
        (unsigned)gx, (unsigned)gy, 1,
        (unsigned)bx, (unsigned)by, 1,
        0,                   // sharedMemBytes
        cuStream,            // stream
        params,
        nullptr              // extra
    );
    if (cr != CUDA_SUCCESS) {
        LUCHS_LOG_HOST("[REPL/COLOR] cuLaunchKernel failed code=%d name=%s str=%s -> fallback",
                       (int)cr, cu_err(cr), cu_str(cr));
        return false;
    }
    // Optional: do NOT sync here; respect caller's stream semantics.
    auto tL1 = std::chrono::high_resolution_clock::now();

    LUCHS_LOG_HOST("[REPL/COLOR] launch ok grid=%dx%d block=%dx%d ms=%.3f",
                   gx, gy, bx, by, to_ms(tL1 - tL0));

    return true;
#endif // OTTER_USE_NVRTC
}

} // namespace ColoringNVRTC
