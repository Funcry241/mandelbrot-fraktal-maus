// Datei: src/luchs_log_device.hpp
// 🐭 Maus-Kommentar: Nur für __device__-Code. Kein __CUDA_ARCH__-Branching, sondern bewusst selektiv eingebunden.
// 🦦 Otter: Rückkehr zu stabiler Einfachheit - keine Formatierung, nur Klartext.
// 🦊 Schneefuchs: Sicher auf allen Architekturen, kein undefined behavior.

// Datei: src/luchs_log_device.hpp
#pragma once
#include "luchs_cuda_log_buffer.hpp"

// =========================================================================
// 🚀 LUCHS_LOG_DEVICE für __device__-Code
//     Nur ein Argument erlaubt. Jeder Aufruf mit >1 Argumenten
//     schlägt schon beim Präprozessor fehl.
// =========================================================================
#define LUCHS_LOG_DEVICE(msg) \
    LuchsLogger::deviceLog(__FILE__, __LINE__, (msg))

