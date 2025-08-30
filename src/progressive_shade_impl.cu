///// Otter: Progressive-Shading (impl)
///// Schneefuchs: Saubere Trennung, nur Host-seitige Logs.
///// Maus: Keine Format-Logs im Devicepfad.
#include "progressive_shade.cuh"
#include "progressive_iteration.cuh"
#include "luchs_log_host.hpp"

namespace prog {

static inline dim3 chooseBlock() { return dim3(32, 8, 1); }
static inline dim3 chooseGrid(int w, int h, dim3 b) {
    return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y, 1);
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

    const dim3 b = chooseBlock();
    const dim3 g = chooseGrid(w, h, b);

    k_shade_progressive_rgba<<<g, b, 0, stream>>>(
        s.dFlags(), s.dIterations(), s.dEscapeIter(), s.dZ(),
        d_out, w, h, maxIterCap
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace prog
