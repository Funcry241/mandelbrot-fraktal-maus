// Datei: src/luchs_log_device.hpp
// 🐭 Maus-Kommentar: Nur für __device__-Code. Kein __CUDA_ARCH__-Branching, sondern bewusst selektiv eingebunden.

#pragma once
#include "luchs_cuda_log_buffer.hpp"

#define LUCHS_LOG_DEVICE(msg) LuchsLogger::deviceLog(__FILE__, __LINE__, msg)
