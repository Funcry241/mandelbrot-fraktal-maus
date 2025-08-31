///// Otter: Progressive Iteration & Resume (header)
///// Schneefuchs: Keine verdeckten Pfade; deterministische Logs.
///// Maus: RAII, SoA-State, API-unverändert.
///// Datei: src/progressive_iteration.cuh

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace prog {

struct ViewportParams {
    double centerX;
    double centerY;
    double scale;
    int    width;
    int    height;
};

struct ProgressiveConfig {
    uint32_t maxIterCap    = 50000;
    uint32_t chunkIter     = 512;
    float    bailout2      = 4.0f;
    bool     resetOnChange = true;
    bool     debugDevice   = false;
};

struct ProgressiveMetrics {
    float     kernel_ms = 0.0f;
    uint32_t  stillActive = 0;
    uint32_t  addIterApplied = 0;
};

class CudaProgressiveState {
public:
    CudaProgressiveState() = default;
    ~CudaProgressiveState();

    CudaProgressiveState(const CudaProgressiveState&) = delete;
    CudaProgressiveState& operator=(const CudaProgressiveState&) = delete;

    void ensure(int width, int height);
    void reset(cudaStream_t stream = 0);
    void maybeResetOnChange(const ViewportParams& vp, bool enableReset, cudaStream_t stream = 0);

    ProgressiveMetrics step(const ViewportParams& vp, const ProgressiveConfig& cfg, cudaStream_t stream = 0);

    // --- Zugriff für Shading/Analyse (read-only) ---
    const float2*   dZ()           const { return d_z_; }
    const uint32_t* dIterations()  const { return d_it_; }
    const uint8_t*  dFlags()       const { return d_flags_; }
    const uint32_t* dEscapeIter()  const { return d_escapeIter_; }

    int width()  const { return width_; }
    int height() const { return height_; }

private:
    float2*   d_z_ = nullptr;
    uint32_t* d_it_ = nullptr;
    uint8_t*  d_flags_ = nullptr;
    uint32_t* d_escapeIter_ = nullptr;
    uint32_t* d_activeCount_ = nullptr;

    int width_ = 0;
    int height_ = 0;

    double lastCx_ = 0.0, lastCy_ = 0.0, lastScale_ = 0.0;
};

} // namespace prog
