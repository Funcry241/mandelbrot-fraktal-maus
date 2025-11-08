///// Otter: Nacktmull — NVRTC coloring runtime header (lightweight); returns true if JIT path used
///// Schneefuchs: Stabile API; keine schweren Includes; /WX-safe; ASCII-Logs
///// Maus: Aktiv nur bei OTTER_USE_NVRTC && Settings::Luchs::{enabled,nvrtc}; sonst Fallback
///// Datei: src/coloring_runtime_nvrtc.hpp

#pragma once
#include <stdint.h>
#include <cuda_runtime.h> // for uchar4, cudaStream_t

namespace ColoringNVRTC {

    // Startet (falls aktiv) den NVRTC-JIT-Coloring-Kernel.
    // Rückgabe: true  => NVRTC-Pfad wurde genutzt (Kernel gestartet)
    //           false => Fallback im Aufrufer verwenden (kein NVRTC)
    bool launch_if_active(const uint16_t* d_it,
                          uchar4*         d_out,
                          int             w,
                          int             h,
                          int             maxIter,
                          cudaStream_t    stream);

} // namespace ColoringNVRTC
