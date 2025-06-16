// Datei: src/memory_utils.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>   // ✨ Fix: für std::exit()
#include "memory_utils.hpp"

namespace MemoryUtils {

// Device-Speicher freigeben (idempotent)
void freeComplexityBuffer(float*& d_buffer) {
    if (d_buffer) {
        cudaFree(d_buffer);  // Fehler werden bewusst ignoriert
        d_buffer = nullptr;
    }
}

} // namespace MemoryUtils
