///// Otter: Clean render-only API — no legacy EC params, no deprecated wrappers; zero host sync; optional done-event.
///// Schneefuchs: Single path (Capybara). Stable signature. Headers/sources stay in sync. ASCII-only logs elsewhere.
///// Maus: Call from cuda_interop via capy_render(...); downstream colorizer writes PBO; no compat layers left.
///// Datei: src/capybara_frame_pipeline.cuh

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#include "luchs_log_host.hpp"
#include "capybara_selector.cuh"   // launch_mandelbrot_select(..., /*useCapybara=*/true)

// -----------------------------------------------------------------------------
// Render iterations into d_it via Capybara (single path).
// No entropy/contrast, no tileSize, no compat wrappers.
// -----------------------------------------------------------------------------
static inline void capy_render(
    uint16_t*     d_it,           // [out] w*h uint16_t iteration buffer (device)
    int           w,              // image width
    int           h,              // image height
    double        cx,             // complex center X
    double        cy,             // complex center Y
    double        stepX,          // complex pixel step X
    double        stepY,          // complex pixel step Y
    int           maxIter,        // iteration cap
    cudaStream_t  renderStream,   // CUDA stream for render
    cudaEvent_t   doneEvent = nullptr // optional: recorded after render (no host sync)
)
{
    // --- Argument hygiene (minimal, consistent) ---
    if (!d_it || w <= 0 || h <= 0 || maxIter < 0) {
        LUCHS_LOG_HOST("[CAPY-FRAME] invalid-args w=%d h=%d maxIter=%d d_it=%p",
                       w, h, maxIter, (void*)d_it);
        return;
    }

    // --- Render (Capybara only; classic path removed) ---
    launch_mandelbrot_select(
        d_it, w, h, cx, cy, stepX, stepY, maxIter, renderStream,
        /*useCapybara=*/true
    );

    // --- Optional event for downstream scheduling (no host sync here) ---
    if (doneEvent != nullptr) {
        const cudaError_t er = cudaEventRecord(doneEvent, renderStream);
        if (er != cudaSuccess) {
            // Deterministic numeric code only (no cudaGetErrorString).
            LUCHS_LOG_HOST("[CAPY-FRAME][ERR] cudaEventRecord rc=%d", (int)er);
        }
    }
}
