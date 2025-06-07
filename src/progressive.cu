// Datei: src/progressive.cu
#include <cstdio>
#include <cmath>
#include <algorithm> // 🐭✨ NEU: nötig für std::min
#include "progressive.hpp"
#include "settings.hpp"

__device__ __managed__ int currentMaxIter = Settings::INITIAL_ITERATIONS;
__device__ __managed__ bool justResetFlag = false;

void resetIterations() {
    currentMaxIter = Settings::INITIAL_ITERATIONS;
    justResetFlag = true;
    if (Settings::debugLogging)
        std::fprintf(stdout, "[RESET] Iterations reset to %d.\n", currentMaxIter);
}

int getCurrentIterations() {
    return currentMaxIter;
}

void incrementIterations() {
    if (currentMaxIter < Settings::MAX_ITERATIONS_CAP) {
        currentMaxIter = std::min(currentMaxIter + Settings::ITERATION_STEP, Settings::MAX_ITERATIONS_CAP);
        if (Settings::debugLogging)
            std::fprintf(stdout, "[UPDATE] Iterations increased to %d.\n", currentMaxIter);
    }
}
