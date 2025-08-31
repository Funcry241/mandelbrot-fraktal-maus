///// Otter: Progressive-Shading (impl)
///// Schneefuchs: Saubere Trennung, nur Host-seitige Logs.
///// Maus: Keine Format-Logs im Devicepfad; optionale Perf-Events nur bei aktivem Logging.
///// Datei: src/progressive_shade_impl.cu

#include "progressive_shade.cuh"
#include "progressive_iteration.cuh"
#include "luchs_log_host.hpp"
#include "settings.hpp"
#include <cuda_runtime.h>

namespace prog {

// konservatives 256-Thread-Shape (gut für viele archs)
static inline dim3 chooseBlock() { return dim3(32, 8, 1); }
static inline dim3 chooseGrid(int w, int h, dim3 b) {
    return dim3((w + int(b.x) - 1) / int(b.x),
                (h + int(b.y) - 1) / int(b.y),
                1);
}

void shade_progressive_to_rgba(const CudaProgressiveState& s,
                               uchar4* d_out,
                               int width, int height,
                               uint32_t maxIterCap,
                               cudaStream_t stream)
{
    // Sicherheitsnetz: Größen müssen zusammenpassen
    if (width != s.width() || height != s.height()) {
        LUCHS_LOG_HOST("[PROG] shade size-mismatch w=%d(have %d) h=%d(have %d)",
                       width, s.width(), height, s.height());
    }
    const int w = s.width();
    const int h = s.height();
    if (w <= 0 || h <= 0) return;

    // Nullprüfungen (rein hostseitige Logs)
    if (!d_out || !s.dFlags() || !s.dIterations() || !s.dEscapeIter() || !s.dZ()) {
        LUCHS_LOG_HOST("[PROG] shade: null device ptr (out=%p flags=%p it=%p esc=%p z=%p)",
                       (void*)d_out, (void*)s.dFlags(), (void*)s.dIterations(),
                       (void*)s.dEscapeIter(), (void*)s.dZ());
        return;
    }

    const dim3 b = chooseBlock();
    const dim3 g = chooseGrid(w, h, b);

    // Optionales Timing nur bei aktivem Logging, kein Device-Format-Log
    cudaEvent_t ev0{}, ev1{};
    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, stream));
    }

    k_shade_progressive_rgba<<<g, b, 0, stream>>>(
        s.dFlags(), s.dIterations(), s.dEscapeIter(), s.dZ(),
        d_out, w, h, maxIterCap
    );
    cudaError_t launchErr = cudaGetLastError();

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        if constexpr (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[PERF] prog_shade %.3f ms grid=(%d,%d) block=(%d,%d)",
                           ms, g.x, g.y, b.x, b.y);
        } else {
            LUCHS_LOG_HOST("[TIME] prog_shade %.3f ms", ms);
        }
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
    }

    // Fehler erst nach Timing prüfen (sofern aktiv)
    CUDA_CHECK(launchErr);
}

} // namespace prog
