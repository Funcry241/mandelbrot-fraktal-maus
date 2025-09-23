///// Otter: Frame pipeline helper slimmed — render-only (Capybara); EC path removed; zero host sync.
///// Schneefuchs: Header-only; ASCII-only LUCHS_LOG_HOST on invalid args; optional cudaEventRecord ecDoneEvent.
///// Maus: Signature kept for source stability; unused EC params ignored; one-path via launch_mandelbrot_select.
///// Datei: src/capybara_frame_pipeline.cuh

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#include "luchs_log_host.hpp"
#include "capybara_selector.cuh"   // launch_mandelbrot_select(...)

// NOTE:
// Legacy entropy/contrast analysis (computeCudaEntropyContrast) has been removed from the codebase.
// This header preserves the former convenience entry point but now performs only the render phase.
// The extra parameters (d_entropy, d_contrast, tileSize) are intentionally ignored.
// If an event is provided, we record it after the render launch on the same stream (no host sync).

static inline void capy_render_and_analyze(
    uint16_t* d_it,               // [out] w*h uint16_t iteration buffer (device)
    float*    d_entropy,          // [out] (ignored) legacy EC buffer
    float*    d_contrast,         // [out] (ignored) legacy EC buffer
    int       w, int h,           // image size
    double    cx, double cy,      // complex center
    double    stepX, double stepY,// complex pixel steps
    int       maxIter,            // iteration cap
    int       tileSize,           // (ignored) legacy EC tile size
    cudaStream_t renderStream,    // CUDA stream for render
    cudaEvent_t  ecDoneEvent,     // optional: recorded after render
    bool      useCapybara = true  // selector flag kept for source stability
)
{
    // --- argument hygiene (minimal, like before) ---
    if (!d_it || w <= 0 || h <= 0 || maxIter < 0 || tileSize <= 0) {
        // Keep the same guard semantics; tileSize retained though unused to avoid silent behavior changes.
        LUCHS_LOG_HOST("[CAPY-FRAME] invalid-args w=%d h=%d maxIter=%d tile=%d d_it=%p",
                       w, h, maxIter, tileSize, (void*)d_it);
        return;
    }

    // Silence unused warnings for removed EC path (we keep signature stable on purpose).
    (void)d_entropy;
    (void)d_contrast;
    (void)tileSize;

    // --- render (Capybara selector; classic path removed inside selector) ---
    launch_mandelbrot_select(d_it, w, h, cx, cy, stepX, stepY, maxIter, renderStream, useCapybara);

    // --- optional event after render (keeps downstream scheduling intact) ---
    if (ecDoneEvent != nullptr) {
        // We deliberately avoid host sync; just record on the given stream.
        CUDA_CHECK(cudaEventRecord(ecDoneEvent, renderStream));
    }
}
