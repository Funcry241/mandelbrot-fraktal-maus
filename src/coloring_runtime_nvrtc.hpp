///// OtterDream — Replikatoren
///// File: src/coloring_runtime_nvrtc.hpp
///// Purpose: NVRTC Coloring Runtime – Austauschbarer color_pixel() (Stub)
///// Phase: 3 (Color-Replikatoren)
///// Hooks: cuda_interop.cu vor colorize_iterations_to_pbo(...)
///// Depends: <stdint.h>, <cuda_runtime.h>
///// Build: /WX-safe ; Header bleibt leicht
///// Log-Tags: [REPL/COLOR]  (Alias: [NVRTC])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Rückgabe true ⇒ NVRTC-Pfad genutzt; false ⇒ Fallback.

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

namespace ColoringNVRTC {

    // Gibt true zurück, wenn ein NVRTC-Shader gestartet wurde (Stub).
    bool launch_if_active(const uint16_t* d_it,
                          uchar4*         d_out,
                          int             w,
                          int             h,
                          int             maxIter,
                          cudaStream_t    stream);

} // namespace ColoringNVRTC
