// Datei: src/luchs_log_device.hpp
// 🐭 Maus-Kommentar: Nur für __device__-Code. Kein __CUDA_ARCH__-Branching, sondern bewusst selektiv eingebunden.
// 🦦 Otter: Rückkehr zu stabiler Einfachheit – keine Formatierung, nur Klartext.
// 🦊 Schneefuchs: Sicher auf allen Architekturen, kein undefined behavior.

#pragma once
#include "luchs_cuda_log_buffer.hpp"

// =========================================================================
// 🚀 LUCHS_LOG_DEVICE für __device__-Code
//     Einfach: nur Klartext-Strings. Keine Formatierung im Kernel.
// =========================================================================
#define LUCHS_LOG_DEVICE(msg) LuchsLogger::deviceLog(__FILE__, __LINE__, msg)
