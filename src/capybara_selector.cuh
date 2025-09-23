///// Otter: One-path selector: classic removed; always dispatch Capybara (Hi/Lo early) with ASCII logs.
///// Schneefuchs: Header-only, zero ABI/API break at call sites; invalid-arg guard; gentle notice if legacy requested.
///// Maus: Same signature as before; bool useCapybara is ignored (kept for source stability).
///// Datei: src/capybara_selector.cuh

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#include "luchs_log_host.hpp"
#include "capybara_api.cuh" // launch_mandelbrot_capybara(...)

// ------------------------------ unified launcher ------------------------------
// Drop-in replacement: keeps the exact same signature as the prior selector.
// The classic fallback has been removed from the project; we always route to Capybara.
static inline void launch_mandelbrot_select(
    uint16_t* d_it,
    int w, int h,
    double cx, double cy,
    double stepX, double stepY,
    int maxIter,
    cudaStream_t stream /*= nullptr*/,
    bool useCapybara /*= CAPY_DEFAULT_ON (ignored)*/
)
{
    // Argument hygiene — keep logs ASCII-only and deterministic.
    if (!d_it || w <= 0 || h <= 0 || maxIter < 0) {
        LUCHS_LOG_HOST("[CAPY-SEL] invalid-args w=%d h=%d maxIter=%d d_it=%p", w, h, maxIter, (void*)d_it);
        return;
    }

    // If a caller explicitly tries to disable Capybara, inform once and proceed.
    // (Classic path is no longer linked into the binary.)
    if (!useCapybara) {
        LUCHS_LOG_HOST("[CAPY-SEL] classic-removed -> forcing capybara");
    }

    // Single rendering path per Genfer Grossente v2.0: immediate use, no silent fallbacks.
    launch_mandelbrot_capybara(d_it, w, h, cx, cy, stepX, stepY, maxIter, stream);
}
