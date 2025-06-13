#pragma once
#ifndef MEMORY_UTILS_HPP
#define MEMORY_UTILS_HPP

#include <cuda_runtime.h>

namespace MemoryUtils {

// 🧠 Allokiert Device-Puffer für Komplexitätsanalyse
float* allocComplexityBuffer(int totalTiles);

// 🧹 Gibt Device-Puffer für Komplexitätsanalyse frei (idempotent)
void freeComplexityBuffer(float*& d_buffer);

} // namespace MemoryUtils

#endif // MEMORY_UTILS_HPP
