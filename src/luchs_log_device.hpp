///// Otter: Device logging shim — route ASCII messages into CUDA device log buffer.
///// Schneefuchs: No varargs on device; only pass preformatted C-strings; host compiles to no-op.
///// Maus: Uses __FILE__/__LINE__ for origin; header-only; safe to include from both host and device code.
///// Datei: src/luchs_log_device.hpp

#pragma once

// Device-side logger macro:
// On CUDA device code, forwards to LuchsLogger::deviceLog(file,line,msg).
// On host-only compilation, compiles to a benign no-op to keep headers shareable.
#if defined(__CUDACC__)
  #include "luchs_cuda_log_buffer.hpp"
  #define LUCHS_LOG_DEVICE(MSG) do { ::LuchsLogger::deviceLog(__FILE__, __LINE__, (MSG)); } while (0)
#else
  // Host fallback: discard to keep single-source headers valid outside NVCC
  #define LUCHS_LOG_DEVICE(MSG) do { (void)sizeof(MSG); } while (0)
#endif
