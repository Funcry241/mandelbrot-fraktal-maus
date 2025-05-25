#pragma once

#include <cuda_runtime.h>

/// Prüft einen CUDA-Fehlercode und bricht bei Fehler mit einer Nachricht ab.
void checkCuda(cudaError_t err, const char* msg);
