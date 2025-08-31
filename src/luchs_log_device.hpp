///// Otter: Nur fuer __device__-Code; Klartext-Logging ohne Formatierung; stabil & simpel.
///// Schneefuchs: Sicher auf allen Architekturen; kein undefined behavior; kein __CUDA_ARCH__-Branching.
///// Maus: Selektiv einbinden; ASCII-only; nutzt LuchsLogger::deviceLog.
///// Datei: src/luchs_log_device.hpp

#pragma once
#include "luchs_cuda_log_buffer.hpp"

// ============================================================================
// LUCHS_LOG_DEVICE fuer __device__-Code
// Nicht-variadisch: jede Verwendung mit >1 Parametern erzeugt einen Compile-Error.
// ============================================================================
#define LUCHS_LOG_DEVICE(msg)                                     \
    do {                                                          \
        LuchsLogger::deviceLog(__FILE__, __LINE__, (msg));        \
    } while (0)
