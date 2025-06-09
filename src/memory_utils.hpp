#pragma once
#ifndef MEMORY_UTILS_HPP
#define MEMORY_UTILS_HPP

#include <cuda_runtime.h>

namespace MemoryUtils {

float* allocComplexityBuffer(int totalTiles);

} // namespace MemoryUtils

#endif // MEMORY_UTILS_HPP
