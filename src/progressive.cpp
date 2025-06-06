// Datei: src/progressive.cpp

#include <cstdio>
#include <cstdlib>   // ✨ Fix: für std::exit()
#include "progressive.hpp"

#ifndef __CUDACC__
int currentMaxIter = 100;  // Host-Version (nur wenn NICHT CUDA-Compiler)
#endif

// 🐭 Reset iterations (only if needed)
void resetIterations() {
    static int lastResetIter = 100;
    if (currentMaxIter != 100) {
        currentMaxIter = 100;
        std::fprintf(stdout, "[RESET] Iterations reset to %d.\n", currentMaxIter);
    }
}

// 🐭 Getter for current iteration count
int getCurrentIterations() {
    return currentMaxIter;
}
