///// OtterDream — Replikatoren
///// File: src/coloring_runtime_nvrtc.cpp
///// Purpose: NVRTC Coloring Runtime – Austauschbarer color_pixel() (Stub)
///// Phase: 3 (Color-Replikatoren)
///// Hooks: cuda_interop.cu vor colorize_iterations_to_pbo(...)
///// Depends: pch.hpp, coloring_runtime_nvrtc.hpp, luchs_log_host.hpp, settings.hpp
///// Build: /WX-safe
///// Log-Tags: [REPL/COLOR]  (Alias: [NVRTC])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Aktiv nur mit OTTER_USE_NVRTC && Settings::Luchs::{enabled,nvrtc}.

#include "pch.hpp"
#include "coloring_runtime_nvrtc.hpp"
#include "luchs_log_host.hpp"
#include "settings.hpp"

namespace ColoringNVRTC {

bool launch_if_active(const uint16_t* d_it,
                      uchar4*         d_out,
                      int             w,
                      int             h,
                      int             maxIter,
                      cudaStream_t    stream)
{
    (void)d_it; (void)d_out; (void)w; (void)h; (void)maxIter; (void)stream;

#if OTTER_USE_NVRTC
    if constexpr (Settings::Luchs::enabled && Settings::Luchs::nvrtc) {
        LUCHS_LOG_HOST("[REPL/COLOR] NVRTC active (stub) → falling back");
        // Hier würde der JIT-Launch erfolgen; wir fallen absichtlich zurück.
        return false; // bewusst Fallback, bis echte Implementierung steht
    }
#endif
    return false; // NVRTC nicht aktiv → Fallback
}

} // namespace ColoringNVRTC
